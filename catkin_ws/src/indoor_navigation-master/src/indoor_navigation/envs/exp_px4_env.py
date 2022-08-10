import gym
import rospy
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from pynput import keyboard
from gym.utils import seeding
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.srv import CommandBool, CommandBoolRequest

from .utils import body_to_world, quaternion_to_euler

class KeyboardController:
    def __init__(self):
        self.key = None
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            self.key = key.char
        except AttributeError:
            self.key = None

    def on_release(self, key):
        self.key = None

    def get_action(self):
        action = np.zeros(4)
        if rospy.get_param("/px4/status") == PX4Status.OFFB_CTRL.value:
            key = self.key
            if key == 'w':
                action[0] = 0.2
            elif key == 'a':
                action[1] = 0.2
            elif key == 's':
                action[0] = -0.2
            elif key == 'd':
                action[1] = -0.2
            elif key == 'h':
                action[3] = 0.2
            elif key == 'j':
                action[2] = -0.2
            elif key == 'k':
                action[2] = 0.2
            elif key == 'l':
                action[3] = -0.2
            elif key == 'q':
                action = None
        return action

class PX4Status(Enum):
    LANDING = 0
    TAKEOFF = 1
    OFFB_CTRL = 2

class ExpPX4Env(gym.Env):
    def __init__(self, reward_mode="dense", default_height=0.8):
        # initialize parameter
        self.scan_init = False
        self.odom_init = False
        self.state_init = False
        self.goal_init = False

        # check environment specification
        self.vehicle_type = rospy.get_param("/px4/vehicle_type")
        if self.vehicle_type == "iris_rplidar":
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

        # private data (experiment parameter)
        self._max_velocity = rospy.get_param("/px4/drone/max_velocity")
        self._min_velocity = rospy.get_param("/px4/drone/min_velocity")
        self._collision_threshold = rospy.get_param("/px4/drone/collision_threshold")
        self._drone_initial_position = np.zeros(3)
        self._default_height = default_height

        self._reach_threshold = rospy.get_param("/px4/goal/reach_threshold")
        self._goal_initial_position = np.zeros(3)

        self._reward_mode = reward_mode
        self._reward_criteria = rospy.get_param("/px4/reward_criteria/"+reward_mode)

        self._px4_state = State()

        # ROS publisher/subscriber/client
        self.odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_cb)
        self.local_vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=10)
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_cb)
        self.goal_pos_sub = rospy.Subscriber("/vrpn_client_node/goal/pose", PoseStamped, self.goal_pos_cb)

        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
        self.set_mode_request = SetModeRequest()
        self.arm_cmd = CommandBoolRequest()
        self._last_request_time = rospy.Time.now()

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
        self.scan_init = False
        self.odom_init = False
        self.state_init = False
        self.goal_init = False
        self._initialize()

        self._landing()
        self._takeoff(0, 0, 0.5)
        state = np.concatenate([self.sensor_data, self.goal_info, self.vel])

        return state

    def step(self, action):
        _goal_distance = self.goal_info[0]
        vx, vy, vz = body_to_world(self.pose[3], action[0], action[1], action[2])
        vz = self._default_height - self.pose[2]
        vel_cmd = np.clip(np.array([vx, vy, vz, action[3]]), self._min_velocity, self._max_velocity)

        self._last_request_time = self._offboard_and_arm(self._last_request_time)
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

    def close(self):
        self._landing()

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

        if (self.odom_init == False):
            self._drone_initial_position[0] = data.pose.pose.position.x
            self._drone_initial_position[1] = data.pose.pose.position.y
            self._drone_initial_position[2] = data.pose.pose.position.z
            self.odom_init = True

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

    def scan_cb(self, data):
        assert len(data.ranges) == self.sensor_data_dim, "[PX4] Laser Scan Dimension Not Matched"

        for i in range(self.sensor_data_dim):
            if data.ranges[i] == float("Inf"):
                self.sensor_data[i] = 6.0
            elif np.isnan(data.ranges[i]):
                self.sensor_data[i] = 0.0
            else:
                self.sensor_data[i] = data.ranges[i]

        if (self.scan_init == False):
            self.scan_init = True

        if min(self.sensor_data) < self._collision_threshold:
            self.collision = True
        else:
            self.collision = False

    def state_cb(self, data):
        self._px4_state = data

        if (self.state_init == False):
            self.state_init = True

    def goal_pos_cb(self, data):
        if (self.goal_init == False):
            self._goal_initial_position[0] = data.pose.position.x
            self._goal_initial_position[1] = data.pose.position.y
            self._goal_initial_position[2] = data.pose.position.z
            self.goal_init = True

    def _takeoff(self, x, y, z):
        self._last_request_time = rospy.Time.now()
        target_position = self._drone_initial_position + np.array([x, y, z])

        print("[PX4] Starting takeoff ...")
        rospy.set_param("/px4/status", PX4Status.TAKEOFF.value)
        while (self._L2norm(target_position, self.pose[:3]) > 0.2):
            self._last_request_time = self._offboard_and_arm(self._last_request_time)

            self.cmd.twist.linear.x = min(max(target_position[0] - self.pose[0], self._min_velocity), self._max_velocity)
            self.cmd.twist.linear.y = min(max(target_position[1] - self.pose[1], self._min_velocity), self._max_velocity)
            self.cmd.twist.linear.z = min(max(target_position[2] - self.pose[2], self._min_velocity), self._max_velocity)
            self.local_vel_pub.publish(self.cmd)
            self.rate.sleep()

            print("delta_x: {:.3f}".format(target_position[0] - self.pose[0]))
            print("delta_y: {:.3f}".format(target_position[1] - self.pose[1]))
            print("delta_z: {:.3f}".format(target_position[2] - self.pose[2]))

        rospy.set_param("/px4/status", PX4Status.OFFB_CTRL.value)

    def _landing(self):
        self._last_request_time = rospy.Time.now()
        target_heights = np.arange(self.pose[2], 0, -0.2)
        target_position = np.array([self.pose[0], self.pose[1], self.pose[2]])

        print("[PX4] Landing ...")
        i = 0
        rospy.set_param("/px4/status", PX4Status.LANDING.value)
        while (self.pose[2] > 0.4 and i < len(target_heights)):
            self._last_request_time = self._offboard_and_arm(self._last_request_time)

            self.cmd.twist.linear.x = min(max(target_position[0] - self.pose[0], self._min_velocity), self._max_velocity)
            self.cmd.twist.linear.y = min(max(target_position[1] - self.pose[1], self._min_velocity), self._max_velocity)
            self.cmd.twist.linear.z = min(max(target_position[2] - self.pose[2], self._min_velocity), self._max_velocity)
            self.local_vel_pub.publish(self.cmd)
            self.rate.sleep()

            print("delta_x: {:.3f}".format(target_position[0] - self.pose[0]))
            print("delta_y: {:.3f}".format(target_position[1] - self.pose[1]))
            print("delta_z: {:.3f}".format(target_position[2] - self.pose[2]))

            if self._L2norm(target_position, self.pose[:3]) < 0.1:
                i += 1
                target_position[2] = target_heights[i]

        self._last_request_time = rospy.Time.now()
        while (self.pose[2] > 0.4 and self._px4_state.armed):
            self._last_request_time = self._auto_land_and_disarm(self._last_request_time)
            self.rate.sleep()

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

    def _initialize(self):
        print("[PX4] Waiting all callback messages received ...")
        while (self.scan_init != True or self.odom_init != True or self.state_init != True or self.goal_init != True):
            self.rate.sleep()

        print("[PX4] Initializing ...")
        self.cmd = TwistStamped()
        self.cmd.twist.linear.z = 0.5
        for i in range(100):
            self.local_vel_pub.publish(self.cmd)
            self.rate.sleep()

        rospy.loginfo("initial_pose: {:.3f}, {:.3f}, {:.3f}".format(self._drone_initial_position[0],
                                                                    self._drone_initial_position[1],
                                                                    self._drone_initial_position[2]))

    def _offboard_and_arm(self, last_request):
        self.set_mode_request.custom_mode = "OFFBOARD"
        self.arm_cmd.value = True

        if (self._px4_state.mode != "OFFBOARD" and rospy.Time.now() - last_request > rospy.Duration(5.0)):
            rospy.wait_for_service("/mavros/set_mode")
            ans = self.set_mode_client(self.set_mode_request)
            if ans.mode_sent:
                rospy.loginfo("Offboard enable")
        else:
            if (not self._px4_state.armed) and (rospy.Time.now() - last_request > rospy.Duration(5.0)):
                rospy.wait_for_service("/mavros/cmd/arming")
                ans = self.arming_client(self.arm_cmd)
                if ans.success:
                    rospy.loginfo("Vehicle armed")

        if rospy.Time.now() - last_request > rospy.Duration(5.0):
            last_request = rospy.Time.now()

        return last_request

    def _auto_land_and_disarm(self, last_request):
        self.set_mode_request.custom_mode = "MANUAL"
        self.arm_cmd.value = False

        if (self._px4_state.mode != "MANUAL" and rospy.Time.now() - last_request > rospy.Duration(5.0)):
            rospy.wait_for_service("/mavros/set_mode")
            ans = self.set_mode_client(self.set_mode_request)
            if ans.mode_sent:
                rospy.loginfo("MANUAL enable")

        if self._px4_state.armed:
            rospy.wait_for_service("/mavros/cmd/arming")
            ans = self.arming_client(self.arm_cmd)
            if ans.success:
                rospy.loginfo("Vehicle disarmed")

        if rospy.Time.now() - last_request > rospy.Duration(5.0):
            last_request = rospy.Time.now()

        return last_request

    def _L2norm(self, pos1, pos2):
        return np.sqrt(np.sum(np.square(pos1-pos2)))

if __name__ == "__main__":
    rospy.init_node("exp_px4_env_node")
    env = ExpPX4Env()

    EPISODE_NUM = 1
    MAX_TIMESTEPS = int(1e7)
    KEYBOARD_CONTROL = True

    # UNCOMMENT THE CODE BELOW TO VISUALIZE THE UAV VELOCITY COMMAND.
    # HOWEVER, THE COMMAND VISUALIZER USES SSH X11 FORWARD, WHICH SLOWS THE CONTROL RATE.
    # UNDER REAL FLIGHT CIRCUMSTANCE, PLEASE KEEP THE FOLLOWING CODE COMMENTED.

    if KEYBOARD_CONTROL:
        # labels = ['Vx', 'Vy', 'Vz', 'W']
        # x = np.arange(len(labels))
        # width = 0.35

        # plt.ion()
        # figure, ax = plt.subplots()
        # command = ax.bar(x, np.zeros(4), width, label='velocity command', color=['red', 'red', 'red', 'red'])
        # ax.set_ylim([-0.3, 0.3])
        # ax.set_xticks(x)
        # ax.set_xticklabels(labels)
        # ax.set_xlabel("Velocity")
        # ax.set_ylabel("m/s")
        # ax.set_title("UAV Velocity Command")
        # ax.legend()

        kc = KeyboardController()

    for ep in range(EPISODE_NUM):
        state = env.reset()
        for t in range(MAX_TIMESTEPS):
            if KEYBOARD_CONTROL:
                action = kc.get_action()
                if action is None:
                    break
            else:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            if done:
                break
            state = next_state

            # if KEYBOARD_CONTROL:
            #     for i in range(4):
            #         command.patches[i].set_height(action[i])
            #     figure.canvas.draw()
            #     figure.canvas.flush_events()

    env.close()
