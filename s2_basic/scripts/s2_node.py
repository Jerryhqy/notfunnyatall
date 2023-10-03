#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

class S2Node(Node):
		# This is an empty class whose parent is Node
    def __init__(self):
				# This function gets called when you initialize your node.
        super().__init__("s2_node")
        self.sub = self.create_subscription(Odometry,"odom", self.odom_callback, 10)
        self.pub = self.create_publisher(Twist,"cmd_vel", 10)
        self.timer = self.create_timer(1,self.vel_callback)
        
    def odom_callback(self,msg):
    	x_value = msg.pose.pose.position.x
    	y_value = msg.pose.pose.position.y
    	print(f"Turtlebot (X,Y)= ({x_value},{y_value})")
    	
    def vel_callback(self):
    	msg = Twist()
    	msg.linear.x = 0.2
    	self.pub.publish(msg)
    	print("sent a command")
    	
    	

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
