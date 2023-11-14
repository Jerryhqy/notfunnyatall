#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from asl_tb3_lib.control import BaseHeadingController
from std_msgs.msg import Bool
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

# the heading controller class
class PerceptionController(BaseHeadingController):
	def __init__(self):
		# initialize using the parent's initialization method
		super().__init__('perception_controller')

		self.image_detected = False

		self.sub = self.create_subscription(Bool, 'detector_bool', self.img_callback, 10)

		# define the proportional control gain and set to 2.0
		# self.kp = 2.0

		self.declare_parameter("kp", 2.0)
		self.declare_parameter("active", self.image_detected)
	
	@property
	def kp(self) -> float:

		return self.get_parameter("kp").value
	
	@property
	def active(self) -> bool:

		return self.get_parameter("active").value
	
	def img_callback(self, msg):
		if msg.data:
			self.image_detected = True
			print('detected')

	
	# def compute_control_with_goal(self, current_state:TurtleBotState, desired_state:TurtleBotState)->TurtleBotControl:

	# 	# compute the heading error (as a wrapped difference)
	# 	hdg_err = wrap_angle(desired_state.theta - current_state.theta)

	# 	# create a new TurtleBotControl message
	# 	control = TurtleBotControl()

	# 	# set omega to the proportional gained value
	# 	control.omega = self.kp * hdg_err

	# 	# return the message
	# 	return control
	
	def compute_control_with_goal(self, current_state:TurtleBotState, desired_state:TurtleBotState)->TurtleBotControl:

		# create a new TurtleBotControl message
		control = TurtleBotControl()

		# set omega to the proportional gained value
		if self.image_detected:
			control.omega = 0.0
		else:
			control.omega = 0.2

		# return the message
		return control


if __name__ == "__main__":
# initialize the ROS2 system
	rclpy.init()
	controller = PerceptionController()
	rclpy.spin(controller)
	rclpy.shutdown()
	
	
