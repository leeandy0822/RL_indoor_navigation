#! /usr/bin/env python
import rospy
import cv2
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import CommandBool, CommandBoolRequest
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.msg import State

class ModeSwitcher:
    def __init__(self):
        self.current_state = State()
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_cb, queue_size=10)
        self.local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

        self.rate = rospy.Rate(20.0)

        self.pose = PoseStamped()
        self.pose.pose.position.x = 0;
        self.pose.pose.position.y = 0;
        self.pose.pose.position.z = 3.0;

        self.offb_set_mode = SetModeRequest()
        self.offb_set_mode.custom_mode = "OFFBOARD"

        self.arm_cmd = CommandBoolRequest()
        self.arm_cmd.value = True

    def state_cb(self, data):
        self.current_state = data

    def check_connection(self):
        while not self.current_state.connected:
            self.rate.sleep()
        return True

    def initialize(self):
        for i in range(100):
            self.local_pos_pub.publish(self.pose)
            self.rate.sleep()
        return True

    def switch_mode(self):
        last_request = rospy.Time.now()

        while True:
            if (self.current_state.mode != "OFFBOARD" and rospy.Time.now() - last_request > rospy.Duration(5.0)):
                self._set_offboard()
                last_request = rospy.Time.now()
            else:
                if (not self.current_state.armed) and (rospy.Time.now() - last_request > rospy.Duration(5.0)):
                    self._arm()
                    last_request = rospy.Time.now()

            self.rate.sleep()

    def _set_offboard(self):
        rospy.wait_for_service("mavros/set_mode")
        ans = self.set_mode_client(self.offb_set_mode)
        if (ans.mode_sent):
            rospy.loginfo("Offboard enabled")

    def _arm(self):
        rospy.wait_for_service("mavros/cmd/arming")
        ans = self.arming_client(self.arm_cmd)
        if (ans.success):
            rospy.loginfo("Vehicle armed")

if __name__ == "__main__":
    rospy.init_node("switch_mode_node")
    ms = ModeSwitcher()
    status = ms.check_connection()
    print("connection status: " + str(status))
    status = ms.initialize()
    print("initialization status: " + str(status))
    ms.switch_mode()
