import rospy
import ros_numpy
import numpy as np
from ..envs.exp_px4_env import ExpPX4Env
from matplotlib import pyplot as plt
from sensor_msgs.msg import LaserScan
import laser_geometry.laser_geometry as lg

class ExpLocalMapGenerator:
    def __init__(self, sensor_dis, map_precision):
        self.vehicle_type = rospy.get_param("/px4/vehicle_type")
        if self.vehicle_type == "iris_rplidar":
            self.pcl_sub = rospy.Subscriber("/laser/scan", LaserScan, self.pcl_cb)
            self.lp = lg.LaserProjection()
        else:
            print(f"[LocalMapGenerator] Unsupported Vehicle Type {self.vehicle_type}")
            exit(1)

        self.map_precision = map_precision
        self.map_resolution = int(2 * sensor_dis / map_precision) + 1
        self.pcl = None
        self._initialize()

    def pcl_cb(self, data):
        if self.vehicle_type == "iris_rplidar":
            pc2_data = self.lp.projectLaser(data)
            xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc2_data)

            Rz = np.array([[ np.cos(np.pi), -np.sin(np.pi), 0],
                           [ np.sin(np.pi),  np.cos(np.pi), 0],
                           [             0,              0, 1]])

            self.pcl = Rz.dot(xyz.transpose()).transpose()

    def _initialize(self):
        rate = rospy.Rate(30)
        while self.pcl is None:
            rate.sleep()

    def generate(self):
        if self.vehicle_type == "iris_rplidar":
            local_map = np.ones((self.map_resolution, self.map_resolution))
            obstacle_pos = self.pcl
            xx = (obstacle_pos[:, 0] * 1 / self.map_precision).astype(np.int32) * -1
            yy = (obstacle_pos[:, 1] * 1 / self.map_precision).astype(np.int32) * -1
            local_map[xx + int(self.map_resolution/2), yy + int(self.map_resolution/2)] = 0.0
            return local_map

if __name__ == "__main__":
    rospy.init_node("exp_local_map_generator_node")
    lmg = ExpLocalMapGenerator(sensor_dis=6.0, map_precision=0.1)
    env = ExpPX4Env()
    state = env.reset()

    local_map = lmg.generate()
    env.close()
    plt.imshow(local_map)
    plt.show()
