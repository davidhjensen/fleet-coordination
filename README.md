# Multi-Agent Pickup and Dropoff Challenge
You have four roombas, a set of goals to collect, and a few shared dropoffs. How fast can you complete the task?
- Off the bat, you need a way to determine distances between points in the map. A Probabalistic Road Map (PRM) does the trick - essentially, throw a bunch of points on the map, and try to connect points to other points in a small neighborhood, discarding edges and points that collide with known obstacles.
- The second step is to assign tasks and dropoff locations to the robots. The optimal solution can be approximated by forming a graph with nodes for all robots and goals that are connected with edge weights determined by the distance to drop off the current goal and move to the start of the next. Running a traveling salesman approximation gives a reasonably balanced assignment, which is then used to generate paths using Dykstra's on the PRM.
- From there, each robot receives its path and an associated priority, which is used in the online collision avoidance. For path following, each robot uses a pure-pursuit controller with an adaptive look-ahead radius and few cool tricks for sharp corners to maximize the speed of maneuvers.
- Finally, online collsion avoidance is acheived with a prioritized approach. When two robots approach each other, the one with lower priority moves directly away while keeping track of the path it follows. It then retraces it steps and continues on its original path.

<table>
  <tr>
    <td style="vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/51098bad-5239-46be-bbeb-5efe7143b316" width="100%">
    </td>
    <td style="vertical-align: bottom;">
      <img width="750" height="750" src="https://github.com/user-attachments/assets/97b0566b-94bd-4dcb-beb7-c006d9bc47fc" width="100%">
    </td>
  </tr>
</table>
