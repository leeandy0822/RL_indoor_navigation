import sys
import rospy
import ros_numpy
import numpy as np
import open3d as o3d
from pynput import keyboard
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
import laser_geometry.laser_geometry as lg
from ..envs.exp_px4_env import ExpPX4Env, PX4Status
from quadprog import solve_qp
import matplotlib.pyplot as plt

class DummyExpPX4Env:
    def __init__(self):
        rospy.set_param("/px4/status", PX4Status.OFFB_CTRL.value)

    def reset(self):
        return None

    def step(self, action):
        return None, None, None, None

    def close(self):
        pass

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

class ExpConstraintGenerator:
    def __init__(self, height_constraint=False, **kwargs):
        self.vehicle_type = rospy.get_param("/px4/vehicle_type")
        if self.vehicle_type == "iris_rplidar":
            self.pcl_sub = rospy.Subscriber("/laser/scan", LaserScan, self.pcl_cb)
            self.lp = lg.LaserProjection()
        else:
            print(f"[ConstraintGenerator] Unsupported Vehicle Type {self.vehicle_type}")
            exit(1)

        self.height_constraint = height_constraint
        if self.height_constraint:
            self.min_height = kwargs.pop("min_height", 0.5)
            self.max_height = kwargs.pop("max_height", 2.0)
            self.odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_cb)
            self.height = 0.0

        self.pcl = None
        self.safe_dis = rospy.get_param("/px4/drone/safety_distance")
        self.max_velocity = rospy.get_param("/px4/drone/max_velocity")
        self.min_velocity = rospy.get_param("/px4/drone/min_velocity")

        self.G = np.identity(3)
        self.a = np.zeros(3)
        self.C = None
        self.b = None

        self._initialize()

    def pcl_cb(self, data):
        if self.vehicle_type == "iris_rplidar":
            pc2_data = self.lp.projectLaser(data)
            xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc2_data)

            Rz = np.array([[ np.cos(np.pi), -np.sin(np.pi), 0],
                           [ np.sin(np.pi),  np.cos(np.pi), 0],
                           [             0,              0, 1]])

            self.pcl = Rz.dot(xyz.transpose()).transpose()

    def odom_cb(self, data):
        self.height = data.pose.pose.position.z

    def constraint_solver(self, u):
        """
        To find a u_star to minimize (||u_star - u||)^2 / 2
        ==> solve quardratic function: (u_star)^2 / 2 - u_star * u

        quadprog solver require the form of quardratic function to
        be like:
                        min     X^T * G * X / 2 - a^T * X
                        s.t     C^T * X >= b

        Our target is to keep the distances between UAV and each detected
        points larger than a safety distance d. Therefore, we designed our h(x) as
                        h(x) = (0 - x)^2 - d^2 >= 0,

        and the derivative of h(x) is
                        h_dot(x) = 2 * (-x) * V_uav.
                        which implys ==> C = (2 * (-x))^T, X = V_uav

        Based on CBF theory, we must find an optimal u_star which satisfy the
        constraint
                        h_dot(x) >= -alpha(h(x))

        Finally, we could obtain the result that
                        G = identity matrix
                        a = V_uav
                        C = (2 * (-x))^T
                        b = -((-x)^2 - d^2)
        """
        u = u.astype(np.double)

        self.a[0] = u[0]
        self.a[1] = u[1]
        self.a[2] = u[2]
        yaw_rate = u[3]

        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(self.pcl)
        downpcl = pcl.voxel_down_sample(voxel_size=0.1)

        pcl = np.asarray(downpcl.points)
        if self.vehicle_type == "iris_depth_camera":
            pcl = pcl[pcl[:, 0] < 5]
            pcl = pcl[np.abs(pcl[:, 1]) < 1.0]
            pcl = pcl[np.abs(pcl[:, 2]) < 0.5]
        elif self.vehicle_type == "iris_rplidar":
            pcl = pcl[np.abs(pcl[:, 0]) < 10.0]
            pcl = pcl[np.abs(pcl[:, 1]) < 10.0]
            pcl = pcl[np.abs(pcl[:, 2]) < 0.5]

        b = -(np.square(-pcl).sum(axis=1) - self.safe_dis**2).reshape(-1)
        C = 2 * (-pcl)

        if self.height_constraint:
            b = np.append(b, np.array([-self.height + self.min_height,
                                       -self.max_height + self.height]))
            C = np.append(C, 1 * np.array([[0, 0,  1],
                                           [0, 0, -1]]), axis=0)

        b = np.append(b, np.array([self.min_velocity,
                                   -self.max_velocity,
                                   self.min_velocity,
                                   -self.max_velocity,
                                   self.min_velocity,
                                   -self.max_velocity]))
        C = np.append(C, 1 * np.array([[ 1,  0,  0],
                                       [-1,  0,  0],
                                       [ 0,  1,  0],
                                       [ 0, -1,  0],
                                       [ 0,  0,  1],
                                       [ 0,  0, -1]]), axis=0)

        self.C = C.transpose()
        self.b = b

        try:
            sol, _, _, _, _, _ = solve_qp(self.G, self.a, self.C, self.b)
            u_star = np.array([sol[0], sol[1], sol[2], yaw_rate])
        except ValueError:
            rospy.logerr("[ConstraintGenerator] No feasible solution satisfies constraint, please execute landing command")
            u_star = np.array([0, 0, 0, 0])

        return u_star

    def height_constraint_solver(self, u):
        assert self.height_constraint == True, "[ConstraintGenerator] Height constraint solver not supported"
        assert self.min_height < self.max_height, "[ConstraintGenerator] Height constraint error (min_height >= max_height)"
        u = u.astype(np.double)
        yaw_rate = u[3]

        self.G = np.identity(3)

        self.a = np.array([u[0], u[1], u[2]])

        self.b = np.array([-self.height + self.min_height,
                           -self.max_height + self.height,
                           self.min_velocity,
                           -self.max_velocity,
                           self.min_velocity,
                           -self.max_velocity,
                           self.min_velocity,
                           -self.max_velocity], dtype=np.double)

        self.C = 1 * np.array([[ 0,  0,  1],
                               [ 0,  0, -1],
                               [ 1,  0,  0],
                               [-1,  0,  0],
                               [ 0,  1,  0],
                               [ 0, -1,  0],
                               [ 0,  0,  1],
                               [ 0,  0, -1]], dtype=np.double).transpose()

        try:
            sol, _, _, _, _, _ = solve_qp(self.G, self.a, self.C, self.b)
            u_star = np.array([sol[0], sol[1], sol[2], yaw_rate])
        except ValueError:
            rospy.logerr("[ConstraintGenerator] No feasible solution satisfies constraint, please execute landing command")
            u_star = np.array([0, 0, 0, 0])

        return u_star

    def _initialize(self):
        rate = rospy.Rate(30)
        while self.pcl is None:
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node("exp_constraint_generator_node")

    desired_pub = rospy.Publisher("/drone/desired_action", TwistStamped, queue_size=10)
    constrained_pub = rospy.Publisher("/drone/constrained_action", TwistStamped, queue_size=10)

    VISUALIZER = False
    HEIGHT_CONSTRAINT = False

    if VISUALIZER:
        labels = ['Vx', 'Vy', 'Vz', 'W']
        x = np.arange(len(labels))
        width = 0.35

        plt.ion()
        figure, ax = plt.subplots()
        desired = ax.bar(x - width/2, np.zeros(4), width, label='desired velocity', color=['red', 'red', 'red', 'red'])
        constrained = ax.bar(x + width/2, np.zeros(4), width, label='constrained velocity', color=['green', 'green', 'green', 'green'])
        ax.set_ylim([-0.3, 0.3])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Velocity")
        ax.set_ylabel("m/s")
        ax.set_title("CBF Control")
        ax.legend()

    kc = KeyboardController()
    env = ExpPX4Env()
    cg = ExpConstraintGenerator(height_constraint=HEIGHT_CONSTRAINT)
    state = env.reset()

    while True:
        desired_action = kc.get_action()
        if desired_action is None:
            break

        constrained_action = cg.constraint_solver(desired_action)
        next_state, reward, done, _ = env.step(constrained_action)
        if done:
            break
        state = next_state

        if VISUALIZER:
            for i in range(4):
                desired.patches[i].set_height(desired_action[i])
                constrained.patches[i].set_height(constrained_action[i])
            figure.canvas.draw()
            figure.canvas.flush_events()

        desired_cmd, constrained_cmd = TwistStamped(), TwistStamped()

        desired_cmd.twist.linear.x = desired_action[0]
        desired_cmd.twist.linear.y = desired_action[1]
        desired_cmd.twist.linear.z = desired_action[2]
        desired_cmd.twist.angular.z = desired_action[3]
        desired_pub.publish(desired_cmd)

        constrained_cmd.twist.linear.x = constrained_action[0]
        constrained_cmd.twist.linear.y = constrained_action[1]
        constrained_cmd.twist.linear.z = constrained_action[2]
        constrained_cmd.twist.angular.z = constrained_action[3]
        constrained_pub.publish(constrained_cmd)

    env.close()
