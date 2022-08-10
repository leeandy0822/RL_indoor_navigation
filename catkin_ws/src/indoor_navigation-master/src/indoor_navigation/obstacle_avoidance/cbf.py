import cv2
import rospy
import ros_numpy
import numpy as np
import open3d as o3d
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, LaserScan
import laser_geometry.laser_geometry as lg
from ..envs.px4_env import PX4Env, PX4Status
from quadprog import solve_qp
import matplotlib.pyplot as plt

def keyboard_control():
    action = np.zeros(4)
    key = cv2.waitKey(1)
    if rospy.get_param("/px4/status") == PX4Status.OFFB_CTRL.value:
        if key == ord('w'):
            action[0] = 0.2
        elif key == ord('a'):
            action[1] = 0.2
        elif key == ord('s'):
            action[0] = -0.2
        elif key == ord('d'):
            action[1] = -0.2
        elif key == ord('h'):
            action[3] = 0.2
        elif key == ord('j'):
            action[2] = -0.2
        elif key == ord('k'):
            action[2] = 0.2
        elif key == ord('l'):
            action[3] = -0.2
        elif key == ord('q'):
            action = None
    return action

class ConstraintGenerator:
    def __init__(self, height_constraint=False, **kwargs):
        self.vehicle_type = rospy.get_param("/px4/vehicle_type")
        if self.vehicle_type == "iris_depth_camera":
            self.pcl_sub = rospy.Subscriber("/camera/depth/points", PointCloud2, self.pcl_cb)
        elif self.vehicle_type == "iris_rplidar":
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
        if self.vehicle_type == "iris_depth_camera":
            camera_xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)

            Rx = np.array([[1,                0,                0],
                           [0,  np.cos(np.pi/2),  np.sin(np.pi/2)],
                           [0, -np.sin(np.pi/2),  np.cos(np.pi/2)]])

            Rz = np.array([[ np.cos(np.pi/2),  np.sin(np.pi/2), 0],
                           [-np.sin(np.pi/2),  np.cos(np.pi/2), 0],
                           [               0,                0, 1]])

            self.pcl = Rz.dot(Rx.dot(camera_xyz.transpose())).transpose()

        elif self.vehicle_type == "iris_rplidar":
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
    rospy.init_node("constraint_generator_node")

    VISUALIZER = True

    if VISUALIZER:
        plt.ion()
        figure, ax = plt.subplots()
        origin = np.zeros(2)
        rotate = ax.plot([0.0],[0.0], marker=r'$\circlearrowleft$', ms=1, color='orange')
        x_axis = ax.plot([-0.3, 0.3], [0.0, 0.0], color='lightgray', ls="--", alpha=0.5)
        y_axis = ax.plot([0.0, 0.0], [-0.3, 0.3], color='lightgray', ls="--", alpha=0.5)
        desired = ax.quiver(origin[0], origin[1], color="r", units="xy", scale=1, label="desired velocity", linewidth=1, alpha=0.7)
        constrained = ax.quiver(origin[0], origin[1], color="g", units="xy", scale=1, label="actual velocity", linewidth=1, alpha=0.5)
        ax.set_xlim([-0.3, 0.3])
        ax.set_ylim([-0.3, 0.3])
        ax.set_xlabel("X-Axis")
        ax.set_ylabel("Y-Axis")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_title("CBF Control")
        ax.legend()

    cv2.imshow("keyboard control", np.zeros((32, 32), np.uint8))
    env = PX4Env()
    cg = ConstraintGenerator(height_constraint=True)
    state = env.reset()

    while True:
        desired_action = keyboard_control()
        if desired_action is None:
            break

        constrained_action = cg.constraint_solver(desired_action)
        next_state, reward, done, _ = env.step(constrained_action)
        if done:
            break
        state = next_state

        if VISUALIZER:
            desired.set_UVC(np.array([-desired_action[0]]), np.array([-desired_action[1]]))
            constrained.set_UVC(np.array([-constrained_action[0]]), np.array([-constrained_action[1]]))
            if constrained_action[3] > 0:
                rotate[0].set_marker(r'$\circlearrowleft$')
                rotate[0].set_markersize(40)
            elif constrained_action[3] < 0:
                rotate[0].set_marker(r'$\circlearrowright$')
                rotate[0].set_markersize(40)
            else:
                rotate[0].set_markersize(0)
            figure.canvas.draw()
            figure.canvas.flush_events()
