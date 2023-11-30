#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from asl_tb3_lib.control import BaseHeadingController
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.grids import snap_to_grid, StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from scipy.signal import convolve2d
    
# the heading controller class
# functionalities
# 1. receive /nav_sucess message from the navigator
# 2. send messages regarding to the
class FrontierExplorer(Node):
    def __init__(self):
        # initialize some variables
        super().__init__("frontier_explorer")
        self.current_state = None
        self.occupancy = None

        # subscribers
        self.nav_success_sub = self.create_subscription(Bool, '/nav_success', self.nav_success_callback, 10)
        self.state_sub = self.create_subscription(TurtleBotState, '/state', self.state_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        # publisher
        self.pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
    ############################### Properties ###################################

    # no need to define properties for this

    ######################### Callback Functions #################################
    # get whether or not the navigation is successful
    def nav_success_callback(self, msg):
        if msg.data:
            msg_out = self.explore()
            self.pub.publish(msg_out)

    # get the current state
    def state_callback(self, msg):
        self.current_state = msg

    # get the occupancy grid
    def map_callback(self, msg: OccupancyGrid) -> None:
        """ Callback triggered when the map is updated

        Args:
            msg (OccupancyGrid): updated map message
        """
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )

        # replan if the new map updates causes collision in the original plan
        # if self.is_planned and not all([self.occupancy.is_free(s) for s in self.plan.path[1:]]):
        #     self.is_planned = False
        #     self.replan(self.goal)

    #################### Implementation Functions #############################
    def explore(self):
        """ returns potential states to explore
        Args:
            occupancy (StochasticOccupancyGrid2D): Represents the known, unknown, occupied, and unoccupied states. See class in first section of notebook.

        Returns:
            frontier_states (np.ndarray): state-vectors in (x, y) coordinates of potential states to explore. Shape is (N, 2), where N is the number of possible states to explore.

        HINTS:
        - Function `convolve2d` may be helpful in producing the number of unknown, and number of occupied states in a window of a specified cell
        - Note the distinction between physical states and grid cells. Most operations can be done on grid cells, and converted to physical states at the end of the function with `occupancy.grid2state()`
        """

        window_size = 13    # defines the window side-length for neighborhood of cells to consider for heuristics
        ########################### Code starts here ###########################
        # index of the center of the kernel
        middle_ind = np.floor(window_size/2).astype(int)

        # for counting ones
        kernel_1 = np.ones([window_size, window_size])/(window_size**2 - 1)
        kernel_1[middle_ind, middle_ind] = 0
        occupied = convolve2d(self.occupancy.probs >= self.occupancy.thresh, kernel_1, mode='valid')
        unknown  = convolve2d(self.occupancy.probs == -1, kernel_1, mode='valid')
        vacant   = 1 - occupied - unknown

        indices = np.where((unknown >= 0.2) & (vacant >= 0.3) & (occupied == 0.0))
        frontier_gridpoints = np.array(indices).T + middle_ind # to account for the sizing window
        frontier_gridpoints[:, [0, 1]] = frontier_gridpoints[:, [1, 0]] # switching columns to get the correct x and y
        frontier_states = self.occupancy.grid2state(frontier_gridpoints)

        # compute the frontier state that is closest to the current state
        distances = np.linalg.norm(frontier_states - np.array([self.current_state.x, self.current_state.y]), axis = 1)
        closest_idx = np.argmin(distances)

        closest_state = frontier_states[closest_idx,:]

        # construct a turtlebotstate object
        target_state = TurtleBotState()

        # specify x and y
        target_state.x = closest_state[0]
        target_state.y = closest_state[1]
        target_state.theta = 0.0

        #print("The closest fronter state to the current state is:")
        #print(frontier_states[closest_idx,:])
        ########################### Code ends here ###########################

        return target_state

if __name__ == "__main__":
    rclpy.init()
    explore = FrontierExplorer()
    rclpy.spin(explore)
    rclpy.shutdown()
