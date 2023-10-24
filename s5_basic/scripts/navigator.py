#!/usr/bin/env python3

import scipy as sp
import numpy as np
import typing as T
from asl_tb3_lib.grids import StochOccupancyGrid2D
import rclpy
from rclpy.node import Node
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class jerryNav(BaseNavigator):
    def __init__(self, node_name: str = "navigator") -> None:
        super().__init__(node_name)
        # proportional gain
        self.kpx = 2.0
        self.kpy = 2.0
        # differential gain
        self.kdx = 2.0
        self.kdy = 2.0
        # velocity threshold
        self.V_PREV_THRES = 0.0001

    def compute_heading_control(self, current_state:TurtleBotState, desired_state:TurtleBotState)->TurtleBotControl:

		# compute the heading error (as a wrapped difference)
        hdg_err = wrap_angle(desired_state.theta - current_state.theta)
		# create a new TurtleBotControl message
        control = TurtleBotControl()
		# set omega to the proportional gained value
        control.omega = self.kpx * hdg_err
		# return the message
        return control
    
    def compute_trajecotry_tracking_control(self, x: float, y: float, th: float, t: float, plan: TrajectoryPlan) -> T.Tuple[float, float]:
        """
        Inputs:
            x,y,th: Current state
            t: Current time
        Outputs:
            V, om: Control actions
        """

        dt = t - self.t_prev

        ########## Code starts here ##########
        # a note to splev function: evaluate B-spline interpolation values 
        state_desired = plan.desired_state(t=t)
        x_d = state_desired.x
        y_d = state_desired.y

        # compute virtual control (PD Controller)
        u1 = xdd_d + self.kpx * (x_d - x) + self.kdx * (xd_d - self.V_prev * np.cos(th))
        u2 = ydd_d + self.kpx * (y_d - x) + self.kdx * (yd_d - self.V_prev * np.cos(th))

        # compute actual input
        V = self.V_prev + dt * (np.cos(th) * u1 + np.sin(th) * u2)
        om = (-np.sin(th) * u1 + np.cos(th) * u2)/self.V_prev

        ########## Code ends here ##########

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        return V, om
    
    def compute_trajectory_plan(self, state: TurtleBotState, goal: TurtleBotState, occupancy: StochOccupancyGrid2D, resolution: float, horizon: float) -> TrajectoryPlan | None:
        x_init = np.array([state.x, state.y])
        x_goal = np.array([goal.x, goal.y])
        # construct an astar problem
        astar = AStar((0, 0), (horizon, horizon), x_init, x_goal, occupancy)
        # solve the problem
        if not astar.solve() or len(astar.path) < 4:  # check whether the solution exist
            print("No path found")
            return None
        else:
            # access solution path
            x = astar.path[:,0]
            y = astar.path[:,1]
            # reset class var for previous velocity and time

            # DUMMY 
            v_desired = 0.15
            spline_alpha = 0.05

            # compute time stamps using constant velocity heuristics
            ds = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
            dt = np.insert(ds,0,0)/v_desired
            ts = np.cumsum(dt)
            
            # generate cubic spline parameters
            path_x_spline = sp.interpolate.splrep(ts, x, s = spline_alpha)
            path_y_spline = sp.interpolate.splrep(ts, y, s = spline_alpha)
    
            return TrajectoryPlan(
                path=astar.path,
                path_x_spline=path_x_spline,
                path_y_spline=path_y_spline,
                duration=ts[-1],
            )

# the AStar solver
class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        return self.occupancy.is_free(x)
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        return np.linalg.norm(np.array(x1) - np.array(x2))
        # raise NotImplementedError("distance not implemented")
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        for ii in [-1,0,1]: # the x loop
            for jj in [-1,0,1]: # the y loop
                if not (ii==0 and jj==0):
                    x_coord = x[0] + ii * self.resolution
                    y_coord = x[1] + jj * self.resolution
                    current_neighbor = self.snap_to_grid((x_coord, y_coord)) # get the tuple
                    if self.is_free(current_neighbor):
                        neighbors.append(current_neighbor)

        # raise NotImplementedError("get_neighbors not implemented")
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        while self.open_set:
            x_current = self.find_best_est_cost_through()
            
            if x_current == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            
            self.open_set.remove(x_current)
            self.closed_set.add(x_current)

            for x_neigh in self.get_neighbors(x_current):
                if x_neigh in self.closed_set:
                    continue
                tentative_cost_to_arrive = self.cost_to_arrive[x_current] + self.distance(x_current, x_neigh)
                if x_neigh not in self.open_set:
                    self.open_set.add(x_neigh)
                elif tentative_cost_to_arrive > self.cost_to_arrive[x_neigh]:
                    continue

                self.came_from[x_neigh] = x_current
                self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                self.est_cost_through[x_neigh] = tentative_cost_to_arrive + self.distance(x_neigh,self.x_goal)

        return False

        # raise NotImplementedError("solve not implemented")
        ########## Code ends here ##########


class DetOccupancyGrid2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic
    """
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        """Verifies that point is not inside any obstacles by some margin"""
        for obs in self.obstacles:
            if x[0] >= obs[0][0] - self.width * .01 and \
               x[0] <= obs[1][0] + self.width * .01 and \
               x[1] >= obs[0][1] - self.height * .01 and \
               x[1] <= obs[1][1] + self.height * .01:
                return False
        return True


if __name__ == "__main__":
    rclpy.init()
    nav = jerryNav()
    rclpy.spin(nav)
    rclpy.shutdown()


