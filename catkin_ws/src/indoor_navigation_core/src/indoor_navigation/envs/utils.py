import sys
import numpy as np
import cv2
from sensor_msgs.msg import Image
import math

def quaternion_to_euler(q):
    """
    Transform quaternion to euler angle.
    The ordering of quaternion elements is [x, y, z, w]
    Inputs:
        q: List
    Returns:
        roll : Float
        pitch: Float
        yaw  : Float
    """
    sinr_cosp = 2 * (q[3] * q[0] + q[1] * q[2])
    cosr_cosp = 1 - 2 * (q[0] * q[0] + q[1] * q[1])
    roll = math.atan2(sinr_cosp, cosr_cosp)

    pitch = 0
    sinp = 2 * (q[3] * q[1] - q[2] * q[0])
    if math.fabs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (q[3] * q[2] + q[0] * q[1])
    cosy_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def euler_to_quaternion(roll, pitch, yaw):
    """
    Transform euler angle to quaternion.
    The ordering of quaternion elements is [x, y, z, w]
    Inputs:
        roll : Float
        pitch: Float
        yaw  : Float
    Returns:
        q: List
    """
    q = []

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q.append(sr * cp * cy - cr * sp * sy)
    q.append(cr * sp * cy + sr * cp * sy)
    q.append(cr * cp * sy - sr * sp * cy)
    q.append(cr * cp * cy + sr * sp * sy)

    return q

def body_to_world(yaw, x, y, z):
    R = np.array([[ np.cos(yaw), -np.sin(yaw), 0],
                  [ np.sin(yaw),  np.cos(yaw), 0],
                  [           0,            0, 1]])
    body_frame = np.array([[x],
                           [y],
                           [z]])
    world_frame = np.dot(R, body_frame)

    return float(world_frame[0, 0]), float(world_frame[1, 0]), float(world_frame[2, 0])

def imgmsg_to_cv2(img_msg):
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    if img_msg.encoding == "rgb8":
        image_opencv = cv2.cvtColor(image_opencv, cv2.COLOR_RGB2BGR)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv

def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg
