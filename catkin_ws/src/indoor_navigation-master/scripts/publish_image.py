#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import sys
sys.path.remove("/opt/ros/melodic/lib/python2.7/dist-packages")
import cv2

def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height
    return img_msg

if __name__ == "__main__":
    rospy.init_node("publish_image_node")
    rate = rospy.Rate(30)
    img_pub = rospy.Publisher("/camera/rgb/image_raw", Image, queue_size=10)
    cap = cv2.VideoCapture(0)

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        img_msg = cv2_to_imgmsg(frame)
        img_pub.publish(img_msg)
        rate.sleep()

    cap.release()
    cv2.destroyAllWindows()
