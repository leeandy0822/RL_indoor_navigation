import cv2
import gym
import time
import rospy
import numpy as np
from enum import Enum
from gym.utils import seeding
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import Empty
from mavros_msgs.msg import State
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ContactsState
from mavros_msgs.srv import SetMode, SetModeRequest
from sensor_msgs.msg import Image, LaserScan

from .utils import body_to_world, quaternion_to_euler, imgmsg_to_cv2

class PX4Status(Enum):
    LANDING = 0
    TAKEOFF = 1
    OFFB_CTRL = 2

class PX4Env(gym.Env):
    def __init__(self, reward_mode="dense"):
        # check environment specification
        self.vehicle_type = rospy.get_param("/px4/vehicle_type")
        if self.vehicle_type == "iris_depth_camera":
            self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_cb)
            self.image_width = rospy.get_param("/px4/drone/sensor_data_dimension/depth_camera/width")
            self.image_height = rospy.get_param("/px4/drone/sensor_data_dimension/depth_camera/height")
            self.sensor_data_dim = self.image_width * self.image_height * 3
        elif self.vehicle_type == "iris_rplidar":
            self.laser_sub = rospy.Subscriber("/laser/scan", LaserScan, self.scan_cb)
            self.sensor_data_dim = rospy.get_param("/px4/drone/sensor_data_dimension/rplidar")
        else:
            print("[PX4] Unsupported Vehicle Type")
            exit(1)
        if reward_mode not in ["dense", "sparse"]:
            print("[PX4] Unsupported Reward Mode")
            exit(1)

        # state variable
        self.cmd = TwistStamped()
        self.pose = np.zeros(4)                 # [x, y, z, yaw]
        self.vel = np.zeros(4)                  # [vx, vy, vz, yaw_rate]
        self.goal_info = np.zeros(2)            # [goal_distance, goal_heading_error]
        self.collision = False
        self.reach = False
        self.sensor_data = np.zeros(self.sensor_data_dim)

        # private data (simulation parameter)
        self._simulation_world = rospy.get_param("/px4/gazebo_world_name")

        self._drone_name = rospy.get_param("/px4/drone/name")
        self._max_velocity = rospy.get_param("/px4/drone/max_velocity")
        self._min_velocity = rospy.get_param("/px4/drone/min_velocity")
        self._drone_candidate_position = np.array(rospy.get_param("/px4/drone/initial_position/"+self._simulation_world))
        self._drone_initial_position = np.zeros(3)

        self._goal_name = rospy.get_param("/px4/goal/name")
        self._reach_threshold = rospy.get_param("/px4/goal/reach_threshold")
        self._goal_candidate_position = np.array(rospy.get_param("/px4/goal/initial_position/"+self._simulation_world))
        self._goal_initial_position = np.zeros(3)

        self._reward_mode = reward_mode
        self._reward_criteria = rospy.get_param("/px4/reward_criteria/"+reward_mode)

        self._px4_state = State()

        # ROS publisher/subscriber/client
        self.odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_cb)
        self.local_vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=10)
        self.collision_sub = rospy.Subscriber("/px4/collision", ContactsState, self.contact_cb)
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_cb)

        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.set_model_proxy = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.set_mode_proxy = rospy.ServiceProxy("mavros/set_mode", SetMode)

        # Gym-Like variable
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.sensor_data_dim+2+4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(self._min_velocity, self._max_velocity, shape=(4,), dtype=np.float32)

        rospy.set_param("/px4/status", PX4Status.LANDING.value)
        self.rate = rospy.Rate(5)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        idx = self.np_random.choice(len(self._drone_candidate_position))
        self._drone_initial_position = self._drone_candidate_position[idx]
        idx = self.np_random.choice(len(self._goal_candidate_position))
        self._goal_initial_position = self._goal_candidate_position[idx]

        self._landing()
        self._reset_world()
        self._move_to(self._drone_name, self._drone_initial_position)
        self._move_to(self._goal_name, self._goal_initial_position)
        self._imu_re_initialize()

        self._takeoff(0, 0, 1.5)
        state = np.concatenate([self.sensor_data, self.goal_info, self.vel])

        return state

    def step(self, action):
        _goal_distance = self.goal_info[0]
        vx, vy, vz = body_to_world(self.pose[3], action[0], action[1], action[2])
        vel_cmd = np.clip(np.array([vx, vy, vz, action[3]]), self._min_velocity, self._max_velocity)
        self.cmd.twist.linear.x = vel_cmd[0]
        self.cmd.twist.linear.y = vel_cmd[1]
        self.cmd.twist.linear.z = vel_cmd[2]
        self.cmd.twist.angular.z = vel_cmd[3]
        self.local_vel_pub.publish(self.cmd)
        self.rate.sleep()

        reward = self._get_reward(vel_cmd, _goal_distance)

        done = False
        if self.collision:
            print("[PX4] \033[91mCollision Happened !\033[0m")
            done = True
        elif self.reach:
            print("[PX4] \033[92mGoal!!\033[0m")
            done = True

        info = None

        next_state = np.concatenate([self.sensor_data, self.goal_info, self.vel])

        return next_state, reward, done, info

    def odom_cb(self, data):
        orientation = [data.pose.pose.orientation.x,
                       data.pose.pose.orientation.y,
                       data.pose.pose.orientation.z,
                       data.pose.pose.orientation.w]
        _, _, yaw = quaternion_to_euler(orientation)

        self.pose[0] = data.pose.pose.position.x
        self.pose[1] = data.pose.pose.position.y
        self.pose[2] = data.pose.pose.position.z
        self.pose[3] = yaw
        self.vel[0] = data.twist.twist.linear.x
        self.vel[1] = data.twist.twist.linear.y
        self.vel[2] = data.twist.twist.linear.z
        self.vel[3] = data.twist.twist.angular.z

        goal_angle = np.arctan2(self._goal_initial_position[1] - self.pose[1],
                                self._goal_initial_position[0] - self.pose[0])
        goal_heading_error = goal_angle - yaw
        if goal_heading_error > np.pi:
            goal_heading_error -= 2 * np.pi
        elif goal_heading_error < -np.pi:
            goal_heading_error += 2 * np.pi

        self.goal_info[0] = self._L2norm(self.pose[:2], self._goal_initial_position[:2])
        self.goal_info[1] = goal_heading_error

        if self.goal_info[0] <= self._reach_threshold:
            self.reach = True
        else:
            self.reach = False

    def contact_cb(self, data):
        contact_num = len(data.states)
        if contact_num != 0:
            self.collision = True
        else:
            self.collision = False

    def image_cb(self, data):
        cv_image = imgmsg_to_cv2(data)
        self.sensor_data = cv2.resize(cv_image, (self.image_width, self.image_height)).reshape(-1)

    def scan_cb(self, data):
        assert len(data.ranges) == self.sensor_data_dim, "[PX4] Laser Scan Dimension Not Matched"

        for i in range(self.sensor_data_dim):
            if data.ranges[i] == float("Inf"):
                self.sensor_data[i] = 6.0
            elif np.isnan(data.ranges[i]):
                self.sensor_data[i] = 0.0
            else:
                self.sensor_data[i] = data.ranges[i]

    def state_cb(self, data):
        self._px4_state = data

    def _takeoff(self, x, y, z):
        target_position = self._drone_initial_position + np.array([x, y, z])

        print("[PX4] Starting takeoff ...")
        start = time.time()
        rospy.set_param("/px4/status", PX4Status.TAKEOFF.value)
        while (self._L2norm(target_position, self.pose[:3]) > 0.1):
            self.cmd.twist.linear.x = (target_position[0] - self.pose[0])
            self.cmd.twist.linear.y = (target_position[1] - self.pose[1])
            self.cmd.twist.linear.z = (target_position[2] - self.pose[2])
            self.local_vel_pub.publish(self.cmd)
            self.rate.sleep()
            if time.time() - start > 180.0:
                print("[PX4] \033[93mTakeoff timeout ...\033[0m")
                self._landing()
                self._reset_world()
                self._move_to(self._drone_name, self._drone_initial_position)
                self._move_to(self._goal_name, self._goal_initial_position)
                self._imu_re_initialize()
                start = time.time()

        rospy.set_param("/px4/status", PX4Status.OFFB_CTRL.value)

    def _L2norm(self, pos1, pos2):
        return np.sqrt(np.sum(np.square(pos1-pos2)))

    def _landing(self):
        print("[PX4] Landing ...")
        rospy.set_param("/px4/status", PX4Status.LANDING.value)
        auto_land_set_mode = SetModeRequest()
        auto_land_set_mode.custom_mode = "AUTO.LAND"

        rospy.wait_for_service("/mavros/set_mode")
        ans = self.set_mode_proxy(auto_land_set_mode)

        start = time.time()
        while (self._px4_state.armed):
            self.rate.sleep()
            if time.time() - start > 60.0:
                print("[PX4] \033[93mLanding timeout ...\033[0m")
                break

    def _reset_world(self):
        print("[PX4] Reseting world ...")
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            exit("\033[91m" + "[ERROR] /gazebo/reset_simulation service call failed" + "\033[0m")

    def _move_to(self, name, pos):
        state_msg = ModelState()
        state_msg.model_name = name
        state_msg.pose.position.x = pos[0]
        state_msg.pose.position.y = pos[1]
        state_msg.pose.position.z = pos[2]
        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            self.set_model_proxy(state_msg)
        except (rospy.ServiceException) as e:
            exit("\033[91m" + "[ERROR] /gazebo/set_model_state service call failed" + "\033[0m")

    def _imu_re_initialize(self):
        print("[PX4] Waiting for imu re-initialization ...")
        start = time.time()
        while (self._L2norm(self._drone_initial_position[:2], self.pose[:2]) > 0.1):
            self.rate.sleep()
            if time.time() - start > 30.0:
                print("[PX4] \033[93mRe-initialization timeout ...\033[0m")
                self._landing()
                self._reset_world()
                self._move_to(self._drone_name, self._drone_initial_position)
                self._move_to(self._goal_name, self._goal_initial_position)
                start = time.time()

    def _get_reward(self, vel_cmd, _goal_distance):
        reward = 0.0
        if self._reward_mode == "dense":
            if self._L2norm(vel_cmd[:3], np.zeros(3)) < self._reward_criteria["coeff"]["epsilon"]:
                position_penalty = self._reward_criteria["no_motion"]
            else:
                position_penalty = 0.0
            orientation_penalty = -np.abs(self.goal_info[1])

            reward = (self._reward_criteria["coeff"]["lambda_p"] * position_penalty +
                      self._reward_criteria["coeff"]["lambda_w"] * orientation_penalty)

            if self.collision:
                reward += self._reward_criteria["collision"]
            elif self.reach:
                reward += self._reward_criteria["reach"]
            else:
                reward += self._reward_criteria["coeff"]["lambda_g"] * (_goal_distance - self.goal_info[0])

        elif self._reward_mode == "sparse":
            if self.collision:
                reward += self._reward_criteria["collision"]
            elif self.reach:
                reward += self._reward_criteria["reach"]

        return reward

if __name__ == "__main__":
    rospy.init_node("px4_env_node")
    env = PX4Env()

    EPISODE_NUM = 5
    MAX_TIMESTEPS = 80

    for ep in range(EPISODE_NUM):
        print(f"Episode: {ep+1}")
        state = env.reset()
        for t in range(MAX_TIMESTEPS):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            if done:
                break
            state = next_state
        print()
