#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

# the heading controller class
class HeadingController(BaseHeadingController):
	def __init__(self):
		# initialize using the parent's initialization method
		super().__init__()
		
		# define the proportional control gain and set to 2.0
		self.kp = 2.0
	
	def compute_control_with_goal(self, current_state:TurtleBotState, desired_state:TurtleBotState)->TurtleBotControl:

		# compute the heading error (as a wrapped difference)
		hdg_err = wrap_angle(desired_state.theta - current_state.theta)

		# create a new TurtleBotControl message
		control = TurtleBotControl()

		# set omega to the proportional gained value
		control.omega = self.kp * hdg_err

		# return the message
		return control

if __name__ == "__main__":
# initialize the ROS2 system
	rclpy.init()
	controller = HeadingController()
	rclpy.spin(controller)
	rclpy.shutdown()
	
	
