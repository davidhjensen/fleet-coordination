from dataclasses import dataclass
from mimetypes import init
import json
from typing import Sequence
from dg_commons import PlayerName
from dg_commons.sim import InitSimGlobalObservations, InitSimObservations, SimObservations
from dg_commons.sim.agents import Agent, GlobalPlanner
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.diff_drive import DiffDriveCommands
from dg_commons.sim.models.diff_drive_structures import DiffDriveGeometry, DiffDriveParameters
from dg_commons.sim.models.obstacles import StaticObstacle
from numpydantic import NDArray
from pydantic import BaseModel
import numpy as np
import networkx as nx
import scipy.optimize
import shapely
import scipy
import scipy.sparse
from scipy.spatial import KDTree as KDT
from scipy.sparse.csgraph import shortest_path as scipy_shortest_path

# if you put an auto import and this crashes I am going to lose my mind and please be lucky with the assignments, come onnnnnnn


class GlobalPlanMessage(BaseModel):
    per_agent_tasks: dict[str, list[tuple[str, str]]]
    global_agent_priorities: dict[str, int]
    per_agent_paths: dict[PlayerName, list[tuple[float, float]]]
    per_agent_flags: dict[PlayerName, list[int]]
    vertices: NDArray
    goals: int


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 10
    # Turn-in-PLace (TiP) CONTROLLER PARAMS
    angular_tol: float = 0.01  # Tolerance for angular position error when turning in place
    pos_tol: float = 0.05  # Tolerance for position to follow next checkpoint
    kp_pos: float = 20.0  # Kp parameter of P controller for position
    kp_lat: float = 2.0  # Kp parameter of P controller for lateral deviation
    kp_omega: float = 5.0  # Kp parameter of P controller for hea
    kp_omega_in_place: float = 2.0
    # Pure - Pursuit (PP) parameters
    window_r: int = 6  # number of sampled points from the path to check circle-line intersections
    look_ahead_d_start: float = 0.5  # look-ahead standard radius for PP controller

    collection_radius: float = 0.3  # Placeholder for goal collection points radius
    eps: float = 1e-2  # tolerance for checking point inclusion in in line-segment intersections
    collection_eps: float = collection_radius / 2  # tolerance for collection

    # ONLINE AVOIDANCE paramaters
    r_agent = 0.6
    obst_eps_buffer = 0.2
    r_priority: float = 2.5  # radius inside which obstacle avoidance is used to stop the lower priority robot
    r_fallback: float = 2  # radius inside which obstacle avoidance is used to reverse the lower priority robot
    r_bad: float = 1.5  # radius inside which obstacle avoidance is used to stop the higher priority robot
    vision_cone: float = (
        np.pi / 2
    )  # the angle from center in which agent detections count (only look for agents +- angle from direction of movement)
    sampling_eps_forward = 0.1  # sampling distance to the lookahead radius away from agent
    sampling_eps_backwards = 0.35  # sampling distance to the lookahead radius towards agent
    reversing_timesteps = 12  # number of timestep to reverse after dropping off all the goals


from pdm4ar.exercises.ex14.utils import (
    to_pi_minus_pi,
    sgn,
    prm,
    priority,
    switch_gears,
    sample_points,
    line_circle_inter,
    valid_point,
)


@dataclass
class RobotInitState:
    x: float
    y: float
    psi: float
    radius: float
    wheelbase: float
    wheelradius: float
    limits: object


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    name: PlayerName
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: DiffDriveGeometry
    sp: DiffDriveParameters

    def __init__(self):
        # feel free to remove/modify the following
        self.params = Pdm4arAgentParams()
        self.use_pp = True
        self.agent_id = None
        self.ang_vel_lims = None

        # Initialize robot drive parameters
        self.current_pose = None
        self.agent_name = None
        self.agent2priority = None
        self.agent_id = None
        self.task = None
        self.path = None
        self.cur_path = None
        self.goal_flags = None

        # DONE: change when scenario extraction is complete
        omega_lims = DiffDriveParameters.default().omega_limits
        self.ang_vel_lims = np.array([omega_lims[0], omega_lims[1]])
        self.wheel_b = DiffDriveGeometry.default().wheelbase
        self.wheel_r = DiffDriveGeometry.default().wheelradius
        self.v_max = 2 * self.ang_vel_lims[1] * self.wheel_r

        # INIT ADDITIONAL CHANGING PARAMS
        self.look_ahead_d = self.params.look_ahead_d_start  # actual look-ahead radius for PP controller
        self.switched = False  # Flag indicating whether the robot is running in reverse mode
        self.intended_dir = False  # Same as switched, but it only flips if we want to move in another direction (not if we are switching to avoid an agent)
        self.end = False

        # INITIALIZE TARGET
        self.current_target_idx = 1  # Used for TiP controller
        self.c_wpt_idx = 0  # used for PP controller - index of current path point

        # LOGGING DATA
        self.debug_path = None
        self.debug_target_pt = np.zeros(2)
        self.log_data = False

        # GRAPH RELATED
        self.vertices: np.ndarray
        self.vertex_tree: KDT
        self.obstacle_tree: shapely.STRtree

        # ONLINE COLLISION
        self.back_to_wpt = False
        self.reversing_positions = []
        self.cnt = 0
        self.delta = np.pi

    def on_episode_init(self, init_sim_obs: InitSimObservations):

        # FOR WHEN THE SCENARIO IS COMPLETELY EXTRACTED
        self.agent_name = init_sim_obs.my_name
        self.agent_id = self.agent_name[7:]
        self.agent_id = int(self.agent_id)

        obstacles = []
        for obstacle in list(init_sim_obs.dg_scenario.static_obstacles):
            obstacles.append(obstacle.shape.buffer(self.params.r_agent + self.params.obst_eps_buffer))
        self.obstacle_tree = shapely.STRtree(obstacles)

    def on_receive_global_plan(
        self,
        serialized_msg: str,
    ):
        global_plan = GlobalPlanMessage.model_validate_json(serialized_msg)

        if self.agent_name is not None:
            self.agent2priority = global_plan.global_agent_priorities
            self.task = global_plan.per_agent_tasks[self.agent_name]
            self.path = global_plan.per_agent_paths[self.agent_name]
            self.cur_path = self.path
            self.goal_flags = global_plan.per_agent_flags[self.agent_name]
            self.vertices = global_plan.vertices
            self.vertex_tree = KDT(self.vertices)
            self.goal_len = global_plan.goals
            if self.path:
                self.last_target = self.path[self.c_wpt_idx + 1]
        else:
            print("NOT INITIALIZED YET")

    def get_commands(self, sim_obs: SimObservations) -> DiffDriveCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.
        :param sim_obs:
        :return:
        """
        # IF PATH LIST IS EMPTY, STAY STILL
        if not self.path:
            return DiffDriveCommands(omega_l=0.0, omega_r=0.0)

        # GET CURRENT POSE
        if self.agent_name is not None:
            state = sim_obs.players[self.agent_name].state
        else:  # Fallback
            return DiffDriveCommands(omega_l=0.0, omega_r=0.0)

        self.current_pose = np.array([state.x, state.y, state.psi])

        if abs(self.delta) > 0.05:
            self.use_pp = False
        else:
            self.use_pp = True

        # ONLINE COLLISION CHECKING
        collision_warning, ray_to_other = self.check_collision_online(sim_obs, current_pose=self.current_pose)

        if collision_warning == 2:
            # MANDATORY REVERSE BY THE AGENT WITH LOWEST PRIORITY BECAUSE OF SMALL DISTANCE
            if ray_to_other is not None:
                ang_vel = self.reverse_controller(ray_to_other)
            self.reversing_positions.append(self.current_pose[:2])
            return DiffDriveCommands(omega_l=ang_vel[0], omega_r=ang_vel[1])

        elif collision_warning == 1:  # MANDATORY STOP BECAUSE OTHER AGENT IN RANGE OF SIGHT
            return DiffDriveCommands(omega_l=0.0, omega_r=0.0)

        else:
            # RESUMING ACTION WITH THE NORMAL CONTROLLER AND BRINGING THE ROBOT AGAIN TO ITS DESIGNATED PATH
            if self.reversing_positions:
                dist_to_wpt = np.linalg.norm(self.path[self.c_wpt_idx] - self.current_pose[:2])
                if dist_to_wpt > self.params.look_ahead_d_start:
                    self.back_to_wpt = True
                else:
                    self.reversing_positions = []
                    self.back_to_wpt = False
            else:
                self.back_to_wpt = False

        if self.back_to_wpt:
            ang_vel = self.resuming_path_controller(self.reversing_positions, current_pos=self.current_pose[:2])
        elif (self.use_pp) and (collision_warning != 1) and (collision_warning != 2) and (not self.end):
            ang_vel = self.pp_controller(path=self.path)
        elif not self.end:
            ang_vel = self.turn_in_place_controller(path=self.path)
        else:
            pass

        if self.end:
            if self.cnt < self.params.reversing_timesteps:
                heading = self.current_pose[2] + self.intended_dir * np.pi
                heading = to_pi_minus_pi(heading)
                ray_to_other = np.array([np.cos(heading), np.sin(heading)])
                ang_vel = self.reverse_controller(ray_to_other)
                self.cnt += 1
                return DiffDriveCommands(omega_l=0.75 * ang_vel[0], omega_r=0.75 * ang_vel[1])
            else:
                return DiffDriveCommands(omega_l=0.0, omega_r=0.0)

        return DiffDriveCommands(omega_l=ang_vel[0], omega_r=ang_vel[1])

    def check_collision_online(self, sim_obs, current_pose):
        # parse observations and set direction of PP
        other_agents = sim_obs.players.items()

        for name, observation in other_agents:
            if name == self.agent_name:
                carrying_goal = True if observation.collected_goal_id is not None else False

        out = 0
        ray = None
        for name, observation in other_agents:
            if name == self.agent_name:
                continue

            agent_carrying_goal = True if observation.collected_goal_id is not None else False

            # calculate distance between agents
            ray_to_other = np.array([observation.state.x, observation.state.y]) - current_pose[:2]
            dist = np.linalg.norm(ray_to_other)

            # calculate the angle between them
            my_direction = self.current_pose[2] + self.intended_dir * np.pi
            my_direction = to_pi_minus_pi(my_direction)

            direction_of_other = np.arctan2(ray_to_other[1], ray_to_other[0])
            dir_diff = np.abs(my_direction - direction_of_other)
            dir_diff = to_pi_minus_pi(dir_diff)

            c_agent_priority = self.agent2priority[self.agent_name]
            o_agent_priority = self.agent2priority[name]
            pr = priority(carrying_goal, agent_carrying_goal, c_agent_priority, o_agent_priority)

            # mandatory reverse if agents are too close
            if dist < self.params.r_bad and pr:
                if out != 2:
                    out = 2
                    ray = ray_to_other

            # mandatory reverse if agents are too close
            if dist < self.params.r_fallback and not pr:
                return 2, ray_to_other

            # priority-based reverse only if it is in cone of direction of movement
            if dist < self.params.r_priority and not pr and dir_diff < Pdm4arAgentParams.vision_cone:
                if out != 2:
                    out = 1
                    ray = ray_to_other
            # reverse if got to the end
            if dist < self.params.r_priority and pr and self.end:
                if out != 2:
                    out = 2
                    ray = ray_to_other

        return out, ray

    # SIMPLE PD HEADING CONTROLLER:

    def turn_in_place_controller(self, path) -> np.ndarray:
        """
        Apply a velocity that depends on the distance (P controller) and turn in place to match orientation.
        :param path: list of waypoints to follow
        :returns angular velocity commands for the robot
        """

        target_pt = np.array(path[self.current_target_idx])
        if self.current_pose is not None:
            current_pos = self.current_pose[0:2]
            current_psi = to_pi_minus_pi(self.current_pose[2])
        else:  # Fallback
            print("NO POSE AVAILABLE")
            return np.zeros(2)

        p_to_t_vec = target_pt - current_pos

        # Linear distance to target
        dx = np.linalg.norm(p_to_t_vec)

        # desired angle from (last to current target)
        last_t = path[self.current_target_idx - 1]
        angle_vec = target_pt - last_t
        theta = np.arctan2(angle_vec[1], angle_vec[0])
        delta = theta - current_psi
        self.delta = to_pi_minus_pi(delta)

        # HEADING CONTROLLER
        dtheta = self.params.kp_omega * delta

        # SET WHEEL SPEED FOR ANGULAR MOVEMENT
        omega = (self.wheel_b / (2 * self.wheel_r)) * dtheta
        # return early if the delta is too much: idea is to rotate in place first, then go straight.
        if abs(delta) > self.params.angular_tol:
            return np.array([-omega, +omega])

        # LINEAR VELOCITY CONTROLLER
        if dx < self.params.pos_tol:
            # Update index
            self.current_target_idx += 1
            if self.current_target_idx == len(path):
                self.current_target_idx = len(path) - 1
                return np.array([0.0, 0.0])

        linear_vel = self.params.kp_pos * dx
        u = linear_vel / self.wheel_r
        u -= abs(omega)
        # Combine the two components clip to the maximum allowed speed maintaining the same dynamics
        omega_1 = u - omega
        omega_2 = u + omega

        return np.array([omega_1, omega_2])

    # FOR PURE PURSUIT CONTROLLER

    def pp_controller(self, path) -> np.ndarray:
        """
        Apply a velocity command that depends on the heading difference from a lookahead candidate point
        :param path: list of waypoints to follow
        :returns angular velocity commands for the robot
        """

        if self.current_pose is not None:
            current_pos = self.current_pose[0:2]
            current_psi = to_pi_minus_pi(self.current_pose[2])
        else:
            print("No pose available")
            return np.zeros(2)  # Fallback

        if self.switched:
            current_psi += np.pi
            current_psi = to_pi_minus_pi(current_psi)

        current_pose = np.array([current_pos[0], current_pos[1], current_psi])

        candidates = line_circle_inter(path, current_pos, self.c_wpt_idx, self.look_ahead_d)  # FIND CANDIDATE POINTS
        la_point = valid_point(candidates, current_pose, self.last_target)  # SELECT FORWARD FACING POINTS
        self.last_target = la_point
        ang_vel = self.get_angular_vel(la_point, path, current_pose)  # SET ANGULAR VELOCITY

        # LOGGING FOR DEBUGGING
        if self.log_data:
            self.debug_target_pt = la_point

        return ang_vel

    def get_angular_vel(self, target_pt, path, current_pose):
        """
        Computes angular velocities for both wheels based on the current position and position of the target

        :param target_pt: look-ahead point for pure pursuit
        :param path: path to follow
        :param current_pose: current pose (x,y,psi) of the robot
        """

        current_pos = current_pose[0:2]
        current_psi = to_pi_minus_pi(current_pose[2])

        p_to_t_vec = target_pt - current_pos
        theta = np.arctan2(p_to_t_vec[1], p_to_t_vec[0])
        delta = theta - current_psi
        delta = to_pi_minus_pi(delta)

        # UPDATING WAYPOINT DISTANCE
        dx = np.linalg.norm(path[self.c_wpt_idx + 1] - current_pos)

        # CHECK IF NEXT WAYPOINT IS A GOAL
        if self.goal_flags[self.c_wpt_idx + 1]:
            # REDUCE LOOKAHEAD RADIUS TO BE SURE TO COLLECT
            if dx < self.params.look_ahead_d_start + self.params.collection_eps:
                self.look_ahead_d = self.params.collection_radius - self.params.collection_eps

        if dx < self.look_ahead_d:
            # Update index
            self.c_wpt_idx += 1
            self.look_ahead_d = self.params.look_ahead_d_start
            if self.c_wpt_idx == (len(path) - 1):
                self.end = True
                return np.array([0.0, 0.0])
            # Check if the robot can switch gears: do it when the angle difference is > pi/2
            if abs(delta) > (np.pi / 2):
                self.switched = 1 - self.switched
                self.intended_dir = self.switched
                # Recalculate delta
                delta = to_pi_minus_pi(delta - np.pi)

        # HEADING CONTROLLER AND SET WHEEL SPEED FOR ANGULAR MOVEMENT
        u = self.v_max / self.wheel_r
        dtheta = self.params.kp_omega * delta
        omega = (self.wheel_b / (2 * self.wheel_r)) * dtheta

        # Combine the two components clipping to max value, but keeping dynamics
        u -= abs(omega)
        omega_1 = u - omega
        omega_2 = u + omega

        omega_1, omega_2 = switch_gears(self.switched, omega_1, omega_2)

        return np.array([omega_1, omega_2])

    def reverse_controller(self, ray_to_other) -> np.ndarray:
        """
        Controller used to input the robot with commands after an imminent collision is found

        :param ray_to_other: vector going from current position to the position of the robot with which collision is happening
        :returns angular velocity commands for the robot
        :rtype: ndarray[float, float]
        """
        if self.switched:
            ##REVERSING MEANS DE-SWITCHING
            current_heading = self.current_pose[2]
        else:
            # REVERSE
            current_heading = to_pi_minus_pi((self.current_pose[2] + np.pi))

        target_heading = np.arctan2(ray_to_other[1], ray_to_other[0]) + np.pi
        target_heading = to_pi_minus_pi(target_heading)

        # candidate_target = self.sample_points(self.current_pose[:2], target_heading)
        candidate_target = sample_points(
            self.current_pose[:2], target_heading, self.vertex_tree, self.obstacle_tree, self.vertices
        )
        omega_1, omega_2 = self.get_inputs(candidate_target, self.current_pose[:2], current_heading)
        # SWITCH SIGN IF NOT SWITCHED IN THE FIRST PLACE
        switch = 1 - self.switched
        omega_1, omega_2 = switch_gears(switch, omega_1, omega_2)

        return np.array([omega_1, omega_2])

    def resuming_path_controller(self, reversing_positions, current_pos) -> np.ndarray:
        """
        Controller used to go back to the previous waypoint on the path, when interrupted by a collision
        checking action

        :param reversing_positions: list of past positions with poses of the reversing path
        :param current_pos: current position of the robot
        :return: angular velocity commands for the robot
        :rtype: ndarray[float, float]
        """
        # CHECK AMONG THOSE THE CLOSEST TO BEING AT A LOOKAHEAD DISTANCE FROM THE ROBOT
        if reversing_positions:
            pts = np.stack(reversing_positions, axis=0)
            dists = np.linalg.norm(pts - current_pos, axis=1)

            # Indices whose distance is in the ring [0.5, 0.7]
            mask = (dists >= self.params.look_ahead_d_start - self.params.sampling_eps_forward) & (
                dists <= self.params.look_ahead_d_start + self.params.sampling_eps_forward
            )
            candidate_idx = np.where(mask)[0]
            # Extract positions
            if len(candidate_idx) > 0:
                # pick the one closest to the target inside the band
                inside_dists = dists[candidate_idx]
                best_inside = candidate_idx[np.argmin(np.abs(inside_dists - self.params.look_ahead_d_start))]
                candidate_target = pts[best_inside]
            else:
                # Fallback → pick point whose distance is closest to the target
                best_idx = np.argmin(np.abs(dists - self.params.look_ahead_d_start))
                candidate_target = pts[best_idx]

        # Fallback
        else:
            print("RAN OUT OF CANDIDATES")
            candidate_target = self.path[self.c_wpt_idx]

        if self.switched:
            current_heading = to_pi_minus_pi((self.current_pose[2] + np.pi))
        else:
            current_heading = self.current_pose[2]

        omega_1, omega_2 = self.get_inputs(candidate_target, current_pos, current_heading)
        omega_1, omega_2 = switch_gears(self.switched, omega_1, omega_2)

        # Remove one position from the reversing positions to signal backtracking
        self.reversing_positions = self.reversing_positions[:-1]

        return np.array([omega_1, omega_2])

    def get_inputs(self, target_pt, current_pos, current_heading):
        """
        Docstring for get_inputs

        :param target_pt: Description
        :param current_pos: Description
        :param current_heading: Description
        """
        # HEADING CONTROLLER
        p_to_t_vec = target_pt - current_pos
        theta = np.arctan2(p_to_t_vec[1], p_to_t_vec[0])
        delta = theta - current_heading
        delta = to_pi_minus_pi(delta)
        dtheta = self.params.kp_omega * delta
        omega = (self.wheel_b / (2 * self.wheel_r)) * dtheta

        # LINEAR VELOCITY CONTROLLER
        u = self.v_max / self.wheel_r
        u -= abs(omega)
        omega_1 = u - omega
        omega_2 = u + omega
        return omega_1, omega_2


class Pdm4arGlobalPlanner(GlobalPlanner):
    """
    This is the Global Planner for PDM4AR
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task
    """

    def __init__(self):
        # agents, goals and dropoffs names in a list
        self.agents: list = []
        self.agent_radius: float
        self.goals: list = []
        self.dropoffs: list = []

        # HELPER SETS
        self.goal_idx_set: set

        # MAP RELATED
        self.static_obstacles: list = []
        self.boundaries: list = []
        self.robots_init_states: dict = {}
        self.deposit_locations: dict = {}
        self.goal_information: dict = {}
        self.obstacles_rtree: shapely.STRtree

        # COLLISION AND PICKUPS RADII
        self.eps_buffer: float = 0.1  # Augment buffer for PP
        self.pickup_r: float = 0.3
        self.thr = 9
        self.swap_factor = 1.5

        # sampler type
        self.sample_evenly: bool = True

        # LOGGING
        self.log_data: bool = False

    def send_plan(self, init_sim_global_obs: InitSimGlobalObservations) -> str:

        # EXTRACT INFO
        self.extract_scenario(init_sim_global_obs)
        graph, vertices_before = prm(
            agent_radius=self.agent_radius,
            boundaries=self.boundaries,
            init_sim_obs=init_sim_global_obs,
            static_obstacles=self.static_obstacles,
            eps_buffer=self.eps_buffer,
            n_samples=5000,
            sample_evenly=self.sample_evenly,
            log_data=self.log_data,
        )
        self.sample_evenly = True
        self.eps_buffer = 0.25
        tasking, path, goal_flags, priorities, vertices = self.tasking(init_sim_global_obs, graph, vertices_before)
        goal_len = len(self.goals)
        global_plan_message = GlobalPlanMessage(
            global_agent_priorities=priorities,
            per_agent_tasks=tasking,
            per_agent_paths=path,
            per_agent_flags=goal_flags,
            vertices=vertices,
            goals=goal_len,
        )

        return global_plan_message.model_dump_json(round_trip=True)

    def extract_scenario(self, init_sim_obs: InitSimGlobalObservations):
        """
        Parses the scenario and gets all obstacles and players names/thing important for global plan
        :param init_sim_obs: observation of the simulation scene
        Initializes: goals, agents and dropoffs names
                and: all agents initial states, all goals initial states, all dropoffs positions, all obstacles in an RTREE
        """

        # ------------GET INITIAL ROBOTS STATES------------ #
        self.agents = list(init_sim_obs.initial_states.keys())
        init_states_vals = list(init_sim_obs.initial_states.values())
        player_obs = list(init_sim_obs.players_obs.values())
        for key, val in zip(self.agents, init_states_vals):
            model_geo = player_obs[0].model_geometry
            self.robots_init_states[key] = RobotInitState(
                x=val.x,
                y=val.y,
                psi=val.psi,
                radius=model_geo.radius,
                wheelbase=model_geo.wheelbase,
                wheelradius=model_geo.wheelradius,
                limits=player_obs[0].model_params.omega_limits,
            )
        self.agent_radius = self.robots_init_states[key].radius

        # ------------GET DROP-oFF POSITIONS------------ #
        self.dropoffs = list(init_sim_obs.collection_points.keys())
        dropoff_infos = list(init_sim_obs.collection_points.values())
        center = np.ndarray((1, 2))
        for dropoff_name, dropoff in zip(self.dropoffs, dropoff_infos):
            poly = dropoff.polygon
            center = np.array([poly.centroid.x, poly.centroid.y])
            radius = np.linalg.norm(center[None, :] - poly.exterior.coords[0], axis=1)
            collected_goals = dropoff.collected_goals  # might be useless
            self.deposit_locations[dropoff_name] = (center, collected_goals, radius)

        # ------------GET GOALS POSITIONS------------ #
        self.goals = list(init_sim_obs.shared_goals.keys())
        goal_values = list(init_sim_obs.shared_goals.values())
        for goal_id, collection in zip(self.goals, goal_values):
            poly = collection.polygon
            center = np.array([poly.centroid.x, poly.centroid.y])
            radius = np.linalg.norm(poly.exterior.coords[0] - center[None, :], axis=1)
            id_collector = collection.collected_by
            # there is also collection time and delivery time, if we want to make things fancy in the implementation
            self.goal_information[goal_id] = {"pos": center, "id_collector": id_collector, "radius": radius}

        # ------------GET INFO FROM dg POSITIONS------------ #
        # from dg we only need the obstacles and it is already given
        self.obstacles_rtree = init_sim_obs.dg_scenario.strtree_obstacles
        self.boundaries = list(init_sim_obs.players_obs.values())[0].dg_scenario.static_obstacles[-1].shape.bounds
        self.static_obstacles = list(init_sim_obs.players_obs.values())[0].dg_scenario.static_obstacles

    def tasking(
        self, init_sim_obs: InitSimGlobalObservations, warehouse_graph: scipy.sparse.csr_array, vertices: np.ndarray
    ) -> tuple[
        dict[PlayerName, list[tuple[PlayerName, PlayerName]]],
        dict[PlayerName, list[tuple[float, float]]],
        dict[PlayerName, list[int]],
        dict[PlayerName, int],
        np.ndarray,
    ]:
        """
        Calculate tasking using a Deterministic Balanced Greedy approach.
        Constructs the task graph and assigns goals to the agent with the
        current lowest workload to minimize total makespan.
        """

        # define agents and corrisponding indices in the scipy graph
        n_agents = len(self.agents)
        agent2idx = dict(zip(self.agents, range(n_agents)))
        agents_set = set(self.agents)

        # define goal  locations and corrisponding indices in the scipy graph
        n_goals = len(self.goals)
        goal2idx = dict(zip(self.goals, range(n_agents, n_agents + n_goals)))
        goals_set = set(self.goals)

        self.goal_idx_set = set(range(n_agents, n_agents + n_goals))

        # define dropoff locations and corrisponding indices in the scipy graph
        n_dropoffs = len(self.dropoffs)
        dropoff2idx = dict(zip(self.dropoffs, range(n_agents + n_goals, n_agents + n_goals + n_dropoffs)))

        # create np adjacency matrix used for computing task assignment
        task_graph_np = np.zeros((n_agents + n_goals, n_agents + n_goals))
        task_graph = nx.DiGraph()

        # add agents and goals as vertices
        task_graph.add_nodes_from([*agents_set, *goals_set])

        # add weighted edges between all vertices
        for u_agent in agents_set:
            u_idx = agent2idx[u_agent]

            # agent-agent edges with weight zero
            for v_agent in agents_set - {u_agent}:
                v_idx = agent2idx[v_agent]
                weight = 0
                task_graph.add_edge(u_agent, v_agent, weight=weight)
                task_graph_np[u_idx, v_idx] = weight

            # agent-task edges with weight corrisponding to distance from agent to starting point of task
            for v_goal in goals_set:
                v_idx = goal2idx[v_goal]
                weight = scipy_shortest_path(
                    warehouse_graph, method="auto", directed=True, indices=[agent2idx[u_agent]]
                )[0, goal2idx[v_goal]]
                task_graph.add_edge(u_agent, v_goal, weight=weight)
                task_graph_np[u_idx, v_idx] = weight

        for u_goal in goals_set:
            u_idx = goal2idx[u_goal]

            # task-agent edges with weight corrisponding to distance to complete task (pickup -> dropoff)
            for v_agent in agents_set:
                v_idx = agent2idx[v_agent]

                # find the optimal dropoff
                shortest_len = np.inf
                shortest_id = self.dropoffs[0]
                for dropoff in self.dropoffs:
                    this_len = scipy_shortest_path(
                        warehouse_graph, method="auto", directed=True, indices=[goal2idx[u_goal]]
                    )[0, dropoff2idx[dropoff]]
                    if this_len < shortest_len:
                        shortest_len = this_len
                        shortest_id = dropoff

                # add edge with label corrisponding to dropoff location
                task_graph.add_edge(u_goal, v_agent, weight=shortest_len, label=shortest_id)
                task_graph_np[u_idx, v_idx] = weight

            # task-task edges with weight corrisponding to distance to complete current task (pickup -> dropoff),
            # plus distance to next task
            for v_goal in goals_set - {u_goal}:
                v_idx = goal2idx[v_goal]

                # find the optimal dropoff
                shortest_len = np.inf
                shortest_id = self.dropoffs[0]
                for dropoff in self.dropoffs:
                    this_len = scipy_shortest_path(
                        warehouse_graph, method="auto", directed=True, indices=[goal2idx[u_goal]]
                    )[0, dropoff2idx[dropoff]]
                    this_len += scipy_shortest_path(
                        warehouse_graph, method="auto", directed=True, indices=[dropoff2idx[dropoff]]
                    )[0, goal2idx[v_goal]]
                    if this_len < shortest_len:
                        shortest_len = this_len
                        shortest_id = dropoff

                # add edge with label corrisponding to dropoff location
                task_graph.add_edge(u_goal, v_goal, weight=shortest_len, label=shortest_id)
                task_graph_np[u_idx, v_idx] = weight

        # save graph
        # nx.write_weighted_edgelist(task_graph, "task_graph_nx.edgelist")

        print("\n************GETTING THE TASKS************")
        # NOTE change this to when uploading on server
        if len(self.goals) > self.thr:
            # Initialize tracking variables
            best_assignments = {agent: [] for agent in agents_set}
            # assign goals to dropoffs
            dropoff2goals = {dropoff: [] for dropoff in self.dropoffs}
            for goal in self.goals:
                idx_goal = goal2idx[goal]
                min_dist = np.inf
                best_dropoff = "dummy_dropoff"
                for dropoff in self.dropoffs:
                    idx_dropoff = dropoff2idx[dropoff]
                    dist = scipy_shortest_path(warehouse_graph, method="auto", directed=True, indices=[idx_goal])[
                        0, idx_dropoff
                    ]
                    if dist < min_dist:
                        min_dist = dist
                        best_dropoff = dropoff

                dropoff2goals[best_dropoff].append(goal)
            # ALREADY RETURNS THE BALANCED TASKS
            best_assignments, agent2priority = self.clustered_assignments(
                warehouse_graph, agent2idx, dropoff2idx, goal2idx, dropoff2goals, task_graph
            )
        else:
            # Get the tasking using the heuristic approach of lowest makespan and distance to goal
            best_assignments, agent2priority = self.heuristic_tasking(
                unassigned_goals=goals_set, agents_set=agents_set, task_graph=task_graph
            )
        print("************************************\n")
        # Add dummy tasks for end-state logic
        for agent in self.agents:
            if len(best_assignments[agent]) == 0:
                loc = init_sim_obs.initial_states[agent]
                circle = shapely.Point(loc.x, loc.y).buffer(1.5)
                cur_obs = init_sim_obs.dg_scenario.static_obstacles
                init_sim_obs.dg_scenario.static_obstacles = cur_obs + (circle,)
                best_assignments[agent].append((agent, str("dummy")))

        # Recalculate PRM with high samples for detailed pathing
        warehouse_graph, vertices = prm(
            agent_radius=self.agent_radius,
            boundaries=self.boundaries,
            init_sim_obs=init_sim_obs,
            static_obstacles=self.static_obstacles,
            eps_buffer=self.eps_buffer,
            n_samples=15000,
            sample_evenly=self.sample_evenly,
            log_data=self.log_data,
        )

        print(best_assignments)
        paths_dict: dict[PlayerName, list[tuple[float, float]]] = {}
        flags_dict: dict[PlayerName, list[int]] = {}
        paths_dict, flags_dict = self.get_paths_flags(
            best_assignments, agent2idx, goal2idx, dropoff2idx, vertices, warehouse_graph
        )

        print("\n**********TASK ASSIGNMENTS**********")
        for agent in self.agents:
            route = best_assignments[agent]
            path = []
            dist = 0
            prev = agent
            for stop in route:
                path.append(stop[0])
                path.append(stop[1])
                if prev == stop[0]:
                    continue
                # Note: This distance print might be slightly inaccurate now due to graph changes
                # but functionally irrelevant to the robot operation
                if task_graph.has_edge(prev, stop[0]):
                    dist += task_graph[prev][stop[0]]["weight"]
                prev = stop[0]
            print(f"{agent} - {dist}:\t{' -> '.join(path)}")
        print("************************************\n")

        print("\n************PRIORITIES************")
        print(agent2priority)
        print("************************************\n")

        return best_assignments, paths_dict, flags_dict, agent2priority, vertices

    def clustered_assignments(
        self, warehouse_graph, agent2idx, dropoff2idx, goal2idx, dropoff2goals, task_graph
    ) -> tuple[dict[PlayerName, list[tuple[PlayerName, PlayerName]]], dict[PlayerName, int]]:

        n_agents = len(self.agents)
        n_dropoffs = len(self.dropoffs)
        agent_dropoff_mat = np.full((n_agents, n_dropoffs), np.inf)

        agent2dropoff2goal = {agent: {dropoff: [] for dropoff in self.dropoffs} for agent in self.agents}

        for i, agent in enumerate(self.agents):
            idx_agent = agent2idx[agent]
            for j, dropoff in enumerate(self.dropoffs):
                idx_dropoff = dropoff2idx[dropoff]
                goals_per_dropoff = dropoff2goals[dropoff]
                # if dropoff has goals assigned to it
                if goals_per_dropoff:
                    min_dist = np.inf
                    # find the best goal for the given agent-dropoff combo
                    for goal in goals_per_dropoff:
                        # calculate distance agent->goal->dropoff
                        idx_goal = goal2idx[goal]
                        dist = scipy_shortest_path(warehouse_graph, method="auto", directed=True, indices=[idx_agent])[
                            0, idx_goal
                        ]
                        dist += scipy_shortest_path(warehouse_graph, method="auto", directed=True, indices=[idx_goal])[
                            0, idx_dropoff
                        ]
                        # save if it is current best
                        if dist < min_dist:
                            min_dist = dist
                            agent2dropoff2goal[agent][dropoff] = goal
                    agent_dropoff_mat[i, j] = min_dist
                # if no goals are assigned to a dropoff, the value is the distance to the dropoff
                else:
                    dist = scipy_shortest_path(warehouse_graph, method="auto", directed=True, indices=[idx_agent])[
                        0, idx_dropoff
                    ]
                    agent_dropoff_mat[i, j] = dist
        mat_for_extra_dropoffs = agent_dropoff_mat.copy()

        row_idx, col_idx = scipy.optimize.linear_sum_assignment(agent_dropoff_mat)
        agents2goaldropoff = {
            self.agents[i]: [(agent2dropoff2goal[self.agents[i]][self.dropoffs[j]], self.dropoffs[j])]
            for i, j in zip(row_idx, col_idx)
        }

        unused_dropoffs = set(self.dropoffs) - set([item[0][1] for item in agents2goaldropoff.values()])
        for dropoff in unused_dropoffs:
            # get index of dropoff column in the untouched matrix and extract it
            idx = self.dropoffs.index(dropoff)
            column = mat_for_extra_dropoffs[:, idx]

            # find the best agent
            agent = self.agents[np.argmin(column)]
            goal = agent2dropoff2goal[agent][dropoff]
            agents2goaldropoff[agent].append((goal, dropoff))

        # calculate best assignments
        best_assignments = {agent: [] for agent in self.agents}
        for agent in self.agents:
            task_list = agents2goaldropoff.get(agent, [])
            if len(task_list) == 0:
                best_assignments[agent] = []
            else:
                for first_goal, dropoff in task_list:
                    other_goals_set = set(dropoff2goals[dropoff]) - {first_goal}
                    best_assignments[agent].append((first_goal, dropoff))
                    best_assignments[agent].extend([(goal, dropoff) for goal in other_goals_set])

        # calculate priorities
        agent2dist = {}
        for agent in self.agents:
            route = best_assignments[agent]
            path = []
            dist = 0
            prev = agent
            for stop in route:
                path.append(stop[0])
                path.append(stop[1])
                if prev == stop[0]:
                    continue
                # Note: This distance print might be slightly inaccurate now due to graph changes
                # but functionally irrelevant to the robot operation
                if task_graph.has_edge(prev, stop[0]):
                    dist += task_graph[prev][stop[0]]["weight"]
                prev = stop[0]
            agent2dist[agent] = dist

        agent_dist_list = sorted(agent2dist.items(), key=lambda x: x[1])
        priorities = {PlayerName(item[0]): i for i, item in enumerate(agent_dist_list)}

        best_assignments = self.balance(
            best_assignments, warehouse_graph, agent2idx, dropoff2idx, goal2idx, agents2goaldropoff
        )

        return best_assignments, priorities

    def balance(
        self,
        best_assignments: dict[PlayerName, list[tuple[PlayerName, PlayerName]]],
        warehouse_graph,
        agent2idx: dict,
        dropoff2idx: dict,
        goal2idx: dict,
        agents2goaldropoff: dict,
    ) -> dict[PlayerName, list[tuple[PlayerName, PlayerName]]]:

        for agent1 in self.agents:
            for agent2 in self.agents:
                # if agents are the same, do not compare
                if agent1 == agent2:
                    continue
                agent1_goals = [item[0] for item in best_assignments[agent1]]
                agent2_goals = [item[0] for item in best_assignments[agent2]]

                # if either agent is not assigned goals, this means it is not active
                if len(agent1_goals) == 0 or len(agent2_goals) == 0:
                    continue
                # balance workload
                if len(agent1_goals) > len(agent2_goals) + 1:
                    closest_goal = "dummy"
                    best_cost = np.inf
                    for goal_1 in agent1_goals:
                        # get the minimum distance between the goal and all dropoffs of 1 and 2
                        g2drp = np.inf
                        old_g2drp = np.inf
                        for item in agents2goaldropoff[agent2]:
                            potential_new_drop = scipy_shortest_path(
                                warehouse_graph, method="auto", directed=True, indices=[goal2idx[goal_1]]
                            )[0, dropoff2idx[item[1]]]
                            if potential_new_drop < g2drp:
                                dropoff_to = item[1]
                                g2drp = potential_new_drop

                        for item in agents2goaldropoff[agent1]:
                            old_dist = scipy_shortest_path(
                                warehouse_graph, method="auto", directed=True, indices=[goal2idx[goal_1]]
                            )[0, dropoff2idx[item[1]]]
                            if old_dist < old_g2drp:
                                dropoff_from = item[1]
                                old_g2drp = old_dist

                        if potential_new_drop > self.swap_factor * old_dist:
                            continue
                        # get the cost of the robot to go from
                        robot_cost = scipy_shortest_path(
                            warehouse_graph, method="auto", directed=True, indices=[goal2idx[goal_1]]
                        )[0, agent2idx[agent2]]
                        # upload the closest goal and cost
                        if robot_cost < best_cost:
                            best_cost = robot_cost
                            closest_goal = goal_1
                    if closest_goal != "dummy":
                        print(f"Moving: {closest_goal} from agent: {agent1} to agent: {agent2}")
                        # swap the assignment
                        best_assignments[agent1].remove((closest_goal, dropoff_from))
                        best_assignments[agent2].append((closest_goal, dropoff_to))

                elif len(agent2_goals) > len(agent1_goals) + 1:
                    closest_goal = "dummy"
                    best_cost = np.inf
                    for goal_2 in agent2_goals:
                        # get the minimum distance between the goal and all dropoffs of 1 and 2
                        g2drp = np.inf
                        old_g2drp = np.inf
                        for item in agents2goaldropoff[agent1]:
                            potential_new_drop = scipy_shortest_path(
                                warehouse_graph, method="auto", directed=True, indices=[goal2idx[goal_2]]
                            )[0, dropoff2idx[item[1]]]
                            # update lowest distance
                            if potential_new_drop < g2drp:
                                dropoff_to = item[1]
                                g2drp = potential_new_drop

                        for item in agents2goaldropoff[agent2]:
                            old_dist = scipy_shortest_path(
                                warehouse_graph, method="auto", directed=True, indices=[goal2idx[goal_2]]
                            )[0, dropoff2idx[item[1]]]
                            # update lowest distance
                            if old_dist < old_g2drp:
                                dropoff_from = item[1]
                                old_g2drp = old_dist

                        if potential_new_drop > self.swap_factor * old_dist:
                            continue
                        # get the cost of the robot to go from
                        robot_cost = scipy_shortest_path(
                            warehouse_graph, method="auto", directed=True, indices=[goal2idx[goal_2]]
                        )[0, agent2idx[agent1]]
                        # upload the closest goal and cost
                        if robot_cost < best_cost:
                            best_cost = robot_cost
                            closest_goal = goal_2
                    if closest_goal != "dummy":
                        print(f"Moving: {closest_goal} from agent: {agent2} to agent: {agent1}")
                        best_assignments[agent2].remove((closest_goal, dropoff_from))
                        best_assignments[agent1].append((closest_goal, dropoff_to))
        return best_assignments

    def get_paths_flags(self, best_assignments, agent2idx, goal2idx, dropoff2idx, vertices, warehouse_graph):
        paths_dict: dict[PlayerName, list[tuple[float, float]]] = {}
        flags_dict: dict[PlayerName, list[int]] = {}

        for agent in self.agents:
            pts_listt = []
            pts_listt.append(agent2idx[agent])
            for tup in best_assignments[agent]:
                goal_id, collection_id = tup[0], tup[1]
                if not (collection_id == "dummy"):
                    pickup_idx = goal2idx[goal_id]
                    pts_listt.append(pickup_idx)
                    collection_idx = dropoff2idx[collection_id]
                    pts_listt.append(collection_idx)
                else:
                    pickup_idx = agent2idx[goal_id]
                    pts_listt.append(pickup_idx)

            # IF THERE IS ONLY ONE POINT (START LOCATION)
            if len(pts_listt) == 2:
                # RETURN EMPTY LISTS
                paths_dict[agent] = []
                flags_dict[agent] = []
            else:
                rec_path = self.reconstruct_path(
                    warehouse_graph=warehouse_graph, pts_list=pts_listt, goal2idx=goal2idx, vertices=vertices
                )
                indices = [rec_path[index][0] for index in range(len(rec_path))]

                flags_dict[agent] = [rec_path[index][1] for index in range(len(rec_path))]
                paths_dict[agent] = [tuple(v) for v in vertices[indices]]

        return paths_dict, flags_dict

    def reconstruct_path(self, warehouse_graph, pts_list, goal2idx, vertices):

        agent_goals_idx = pts_list[1::2]

        goals_idx = list(goal2idx.values())
        other_goals_idx = [x for x in goals_idx if x not in agent_goals_idx]
        # disable all indices of goals that are not in goal indices and those nearby by a radius of
        close_idx = []
        vertices_stack = np.stack(vertices)

        for goal in other_goals_idx:
            goal_location = vertices[goal]
            dists = np.linalg.norm(vertices_stack - goal_location, axis=1)
            close_indices = np.where(dists < 2 * self.pickup_r)[0]
            close_idx.extend(close_indices)

        # mask warehouse graph edges and give infinite cost to those edges

        graph = warehouse_graph.copy()

        graph[close_idx, :] = np.inf
        graph[:, close_idx] = np.inf

        full_path = [(pts_list[0], int(0))]
        for u, v in zip(pts_list[:-1], pts_list[1:]):
            dists, predecessors = scipy_shortest_path(
                graph,
                method="auto",
                directed=True,
                return_predecessors=True,
                indices=[u],
            )

            preds = predecessors[0]
            if np.isinf(dists[0, v]):
                print(f"WARNING: no path from {u} to {v}")
                continue
            # backtrack
            if v in self.goal_idx_set:
                tup = (v, int(1))
            else:
                tup = (v, int(0))
            segment = [tup]
            cur = v
            while cur != u:
                cur = preds[cur]
                if cur == -9999:
                    print(f"WARNING: broken predecessor chain from {u} to {v}")
                    segment = []
                    break
                tup = (cur, int(0))
                segment.append(tup)
            if not segment:
                continue
            segment.reverse()
            full_path.extend(segment[1:])

        return full_path

    def heuristic_tasking(
        self, unassigned_goals: set, agents_set: set, task_graph
    ) -> tuple[dict[PlayerName, list[tuple[PlayerName, PlayerName]]], dict[PlayerName, int]]:
        # Initialize tracking variables
        best_assignments = {agent: [] for agent in agents_set}

        # Determine current virtual location of agent in the graph (Starts at Agent Node)
        agent_current_node = {agent: agent for agent in agents_set}

        # Track the actual Goal ID the agent is currently handling (to retrieve dropoff info later)
        agent_holding_goal = {agent: None for agent in agents_set}

        # Track accumulated cost (distance) for load balancing
        agent_costs = {agent: 0.0 for agent in agents_set}

        # Loop until all goals are assigned
        while unassigned_goals:

            # Select agent with the lowest accumulated cost so far
            current_agent = min(agent_costs, key=agent_costs.get)
            current_node = agent_current_node[current_agent]

            # Find closest available goal
            best_goal = None
            min_weight = np.inf

            for goal in unassigned_goals:
                if task_graph.has_edge(current_node, goal):
                    w = task_graph[current_node][goal]["weight"]
                    if w < min_weight:
                        min_weight = w
                        best_goal = goal

            if best_goal:
                # If the agent was previously holding a goal, we are now transitioning FROM that goal
                # TO the new goal. The edge (prev_goal -> new_goal) contains the dropoff for the prev_goal.
                previous_goal = agent_holding_goal[current_agent]

                if previous_goal is not None:
                    # Retrieve dropoff for the PREVIOUS goal from the edge connecting to the CURRENT goal
                    dropoff_id = task_graph[previous_goal][best_goal]["label"]
                    best_assignments[current_agent].append((previous_goal, dropoff_id))

                # Update state
                # print(f"Current agent: {current_agent}, cost to go: {min_weight}")
                agent_costs[current_agent] += min_weight
                agent_current_node[current_agent] = best_goal
                agent_holding_goal[current_agent] = best_goal
                unassigned_goals.remove(best_goal)
            else:
                break

        # The loop finishes when goals run out, but the last assigned goal hasn't been added
        # to the assignment list yet
        for agent in self.agents:
            last_goal = agent_holding_goal[agent]
            if last_goal is not None:
                dropoff_id = task_graph[last_goal][agent]["label"]
                best_assignments[agent].append((last_goal, dropoff_id))

        # CALCULATE PRIORITIES
        agents_prioritized = []
        for agent in self.agents:
            agents_prioritized.append((agent_costs[agent], agent))
        agents_prioritized.sort(key=lambda x: x[0])

        priorities = {PlayerName(item[1]): i for i, item in enumerate(agents_prioritized)}

        # return the best assignments for each robot and the list of the players in ascending order (priorities increases as we go through the list)
        return best_assignments, priorities
