#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class S2Node(Node):
		# This is an empty class whose parent is Node
    def __init__(self):
				# This function gets called when you initialize your node.
        super().__init__("s2_node")


def main(args=None):
		# Initialize your ROS context, you must do this before
		# creating any ROS node.
    rclpy.init(args=args)
		
		# Instantiate your ROS Node
    s2_node = S2Node()

		# Spin your ROS node until CTRL+C is entered in the console.
    rclpy.spin(s2_node)


if __name__ == "__main__":
		# Above conditional is true when this script is executed, but
		# not when it is imported (so that you can reuse code).
    main()
