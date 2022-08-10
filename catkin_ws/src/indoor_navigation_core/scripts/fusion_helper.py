#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped

class FusionHelper:
    def __init__(self, trackerID="rigidbody7"):
        self.host_mocap = PoseStamped()
        self.mocap_pos_pub = rospy.Publisher("/mavros/vision_pose/pose", PoseStamped, queue_size=2)
        self.mocap_pos_sub = rospy.Subscriber(f"/vrpn_client_node/{trackerID}/pose", PoseStamped, self.mocap_pos_cb)

    def mocap_pos_cb(self, data):
        self.host_mocap = data

    def sensor_fusion(self):
        """
        publish pose data from OptiTrack to flight control board which will
        automatically do sensor fusion
        """
        rospy.loginfo("odom: {:.3f} {:.3f} {:.3f}".format(self.host_mocap.pose.position.x,
                                                          self.host_mocap.pose.position.y,
                                                          self.host_mocap.pose.position.z))
        self.mocap_pos_pub.publish(self.host_mocap)

if __name__ == "__main__":
    rospy.init_node("fusion_helper_node")
    fusion_helper = FusionHelper(trackerID="rigidbody7")
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        fusion_helper.sensor_fusion()
        rate.sleep()
