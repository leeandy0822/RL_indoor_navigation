#! /usr/bin/env python3
import rospy
import numpy as np
import time
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState

class Moving:
    def __init__(self):
        self.Box_Target = [-0.5, 0.5]
        self.Cylinder_Target = [0.5, -0.5]
        self.Box_idx = 0
        self.Cylinder_idx = 0
        self.set_model_proxy = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.rate = rospy.Rate(10)
        self.moving()

    def moving(self):
        while not rospy.is_shutdown():
            obstacle1 = ModelState()
            obstacle2 = ModelState()
            model = rospy.wait_for_message('gazebo/model_states', ModelStates)
            for i in range(len(model.name)):
                if model.name[i] == 'unit_box_1':
                    obstacle1.model_name = 'unit_box_1'
                    obstacle1.pose = model.pose[i]
                    if np.abs(obstacle1.pose.position.y - self.Box_Target[self.Box_idx]) < 0.1:
                        self.Box_idx = 1 - self.Box_idx
                    if self.Box_Target[self.Box_idx] > obstacle1.pose.position.y:
                        obstacle1.pose.position.y = obstacle1.pose.position.y + 0.01
                    elif self.Box_Target[self.Box_idx] < obstacle1.pose.position.y:
                        obstacle1.pose.position.y = obstacle1.pose.position.y - 0.01
                    rospy.wait_for_service("/gazebo/set_model_state")
                    try:
                        self.set_model_proxy(obstacle1)
                    except (rospy.ServiceException) as e:
                        pass

                elif model.name[i] == 'unit_cylinder':
                    obstacle2.model_name = 'unit_cylinder'
                    obstacle2.pose = model.pose[i]
                    if np.abs(obstacle2.pose.position.y - self.Cylinder_Target[self.Cylinder_idx]) < 0.1:
                        self.Cylinder_idx = 1 - self.Cylinder_idx
                    if self.Cylinder_Target[self.Cylinder_idx] > obstacle2.pose.position.y:
                        obstacle2.pose.position.y = obstacle2.pose.position.y + 0.01
                    elif self.Cylinder_Target[self.Cylinder_idx] < obstacle2.pose.position.y:
                        obstacle2.pose.position.y = obstacle2.pose.position.y - 0.01
                    rospy.wait_for_service("/gazebo/set_model_state")
                    try:
                        self.set_model_proxy(obstacle2)
                    except (rospy.ServiceException) as e:
                        pass

            self.rate.sleep()

def main():
    rospy.init_node('moving_obstacle')
    moving = Moving()

if __name__ == '__main__':
    main()
