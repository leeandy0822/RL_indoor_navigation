#! /usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan

if __name__ == "__main__":
    rospy.init_node("dummy_laser_node")
    pub = rospy.Publisher("/laser/scan", LaserScan)
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        scan = LaserScan()
        scan.ranges = (np.ones((360,)) * 6.0).tolist()
        pub.publish(scan)
        rate.sleep()

