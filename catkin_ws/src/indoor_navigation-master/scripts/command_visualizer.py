#! /usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import TwistStamped

# Run this script before play the bag file.

class CommandVisualizer:
    def __init__(self):
        plt.ion()
        self.figure, self.ax = plt.subplots()
        origin = np.zeros(2)
        self.rotate = self.ax.plot([0.0],[0.0], marker=r'$\circlearrowleft$', ms=1, color='orange')
        self.x_axis = self.ax.plot([-0.3, 0.3], [0.0, 0.0], color='lightgray', ls="--", alpha=0.5)
        self.y_axis = self.ax.plot([0.0, 0.0], [-0.3, 0.3], color='lightgray', ls="--", alpha=0.5)
        self.desired = self.ax.quiver(origin[0], origin[1], color="r", units="xy", scale=1, label="desired velocity", linewidth=1, alpha=0.7)
        self.constrained = self.ax.quiver(origin[0], origin[1], color="g", units="xy", scale=1, label="actual velocity", linewidth=1, alpha=0.5)
        self.ax.set_xlim([-0.3, 0.3])
        self.ax.set_ylim([-0.3, 0.3])
        self.ax.set_xlabel("X-Axis")
        self.ax.set_ylabel("Y-Axis")
        self.ax.invert_xaxis()
        self.ax.invert_yaxis()
        self.ax.set_title("CBF Control")
        self.ax.legend()

        self.desired_sub = rospy.Subscriber("/drone/desired_action", TwistStamped, self.desired_cb)
        self.constrained_sub = rospy.Subscriber("/drone/constrained_action", TwistStamped, self.constrained_cb)

        self.desired_action = np.zeros(4)
        self.desired_vector = np.zeros(2)
        self.constrained_action = np.zeros(4)
        self.constrained_vector = np.zeros(2)

    def desired_cb(self, msg):
        self.desired_action[0] = msg.twist.linear.x
        self.desired_action[1] = msg.twist.linear.y
        self.desired_action[2] = msg.twist.linear.z
        self.desired_action[3] = msg.twist.angular.z
        self.desired_vector[0] = self.desired_action[0]
        self.desired_vector[1] = self.desired_action[1]

    def constrained_cb(self, msg):
        self.constrained_action[0] = msg.twist.linear.x
        self.constrained_action[1] = msg.twist.linear.y
        self.constrained_action[2] = msg.twist.linear.z
        self.constrained_action[3] = msg.twist.angular.z
        self.constrained_vector[0] = self.constrained_action[0]
        self.constrained_vector[1] = self.constrained_action[1]

    def viz(self):
        self.desired.set_UVC(np.array([-self.desired_vector[0]]), np.array([-self.desired_vector[1]]))
        self.constrained.set_UVC(np.array([-self.constrained_vector[0]]), np.array([-self.constrained_vector[1]]))
        if self.constrained_action[3] > 0:
            self.rotate[0].set_marker(r'$\circlearrowleft$')
            self.rotate[0].set_markersize(40)
        elif self.constrained_action[3] < 0:
            self.rotate[0].set_marker(r'$\circlearrowright$')
            self.rotate[0].set_markersize(40)
        else:
            self.rotate[0].set_markersize(0)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

if __name__ == "__main__":
    rospy.init_node("command_visualizer_node")
    cmd_viz = CommandVisualizer()
    while not rospy.is_shutdown():
        cmd_viz.viz()
