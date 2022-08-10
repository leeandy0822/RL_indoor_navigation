import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import gurobipy as gp
from gurobipy import GRB
from .common.actor import Actor
# from .common.critic import Critic
import cv2
import rospy
import ros_numpy
import numpy as np
import open3d as o3d
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, LaserScan
import laser_geometry.laser_geometry as lg
from ..envs.px4_env import PX4Env, PX4Status
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import time

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.state_layers = nn.Sequential(
                              nn.Linear(state_dim, hidden_dim),
                              nn.ReLU(),
                              nn.Linear(hidden_dim, hidden_dim, bias=False),
                            )

        self.action_layers = nn.Sequential(
                               nn.Linear(action_dim, hidden_dim, bias=False),
                             )

        self.shared_bias = nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
        self.shared_layers = nn.Sequential(
                               nn.ReLU(),
                               nn.Linear(hidden_dim, 1)
                             )

        self._init_weights()

    def forward(self, state, action):
        state_feature = self.state_layers(state)
        action_feature = self.action_layers(action)
        state_action_feature = state_feature + action_feature + self.shared_bias
        q = self.shared_layers(state_action_feature)

        return q

    def _init_weights(self):
        for layer in self.state_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        for layer in self.action_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

class NFWPO:
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256, to_cuda=True, **kwargs):
        self.noise_scale = kwargs.pop("noise_scale", 0.2) * max_action
        self.max_noise = kwargs.pop("max_noise", 0.5) * max_action
        self.action_high = kwargs.pop("action_high", 1)
        self.action_low = kwargs.pop("action_low", -1)
        self.discount = kwargs.pop("discount", 0.99)
        self.update_freq = kwargs.pop("update_freq", 2)
        self.tau = kwargs.pop("tau", 0.005)

        if to_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim=hidden_dim).to(self.device)
        self.critic_1 = Critic(state_dim, action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim=hidden_dim).to(self.device)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.optimizer_critic_1 = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)
        self.optimizer_critic_2 = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.total_iter = 0
        self.actor_loss = 0.0
        self.critic_loss = 0.0

        self.vehicle_type = rospy.get_param("/px4/vehicle_type")
        if self.vehicle_type == "iris_depth_camera":
            self.pcl_sub = rospy.Subscriber("/camera/depth/points", PointCloud2, self.pcl_cb)
        elif self.vehicle_type == "iris_rplidar":
            self.pcl_sub = rospy.Subscriber("/laser/scan", LaserScan, self.pcl_cb)
            self.lp = lg.LaserProjection()
            self.angle_min = -3.1400001049
            self.angle_max = 3.1400001049
            self.angle_increment = 0.0174930356443
            self.time_increment = 0.0
            self.scan_time = 0.0
            self.range_min = 0.20000000298
            self.range_max = 6.0
        else:
            print(f"[ConstraintGenerator] Unsupported Vehicle Type {self.vehicle_type}")
            exit(1)

        self.height_constraint = kwargs.pop("height_constraint_solver", True)
        if self.height_constraint:
            self.min_height = kwargs.pop("min_height", 0.5)
            self.max_height = kwargs.pop("max_height", 2.0)
            self.odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_cb)
            self.height = 0.0

        self.pcl = None
        self.safe_dis = rospy.get_param("/px4/drone/safety_distance")
        self.max_velocity = rospy.get_param("/px4/drone/max_velocity")
        self.min_velocity = rospy.get_param("/px4/drone/min_velocity")

        solvers.options['show_progress'] = False
        self.P = matrix(np.identity(3), tc='d')
        self.Q = matrix(np.zeros(3), tc='d')
        self.G = None
        self.H = None

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

    def state_to_pcl(self, state):
        laser_data = LaserScan()
        laser_data.angle_min = self.angle_min
        laser_data.angle_max = self.angle_max
        laser_data.angle_increment = self.angle_increment
        laser_data.time_increment = self.time_increment
        laser_data.scan_time = self.scan_time
        laser_data.range_min = self.range_min
        laser_data.range_max = self.range_max

        for i in range(360):
            if state[i] == 6.0:
                laser_data.ranges.append(np.inf)
                laser_data.intensities.append(0.0)
            elif state[i] == 0.0:
                laser_data.ranges.append(np.nan)
                laser_data.intensities.append(0.0)
            else:
                laser_data.ranges.append(state[i].cpu().data)
                laser_data.intensities.append(0.0)

        pc2_data = self.lp.projectLaser(laser_data)
        xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc2_data)

        Rz = np.array([[ np.cos(np.pi), -np.sin(np.pi), 0],
                       [ np.sin(np.pi),  np.cos(np.pi), 0],
                       [             0,              0, 1]])

        pcl = Rz.dot(xyz.transpose()).transpose()
        return pcl

    def constraint_solver(self, u, pcl_data=None):
        """
        To find a u_star to minimize (||u_star - u||)^2 / 2
        ==> solve quardratic function: (u_star)^2 / 2 - u_star * u

        CVXOPT solver require the form of quardratic function to
        be like:
                        min     X^T * P * X / 2 + Q^T * X
                        s.t     G * X <= H

        Our target is to keep the distances between UAV and each detected
        points larger than a safety distance d. Therefore, we designed our h(x) as
                        h(x) = (0 - x)^2 - d^2 >= 0,

        and the derivative of h(x) is
                        h_dot(x) = 2 * (-x) * V_uav.
                        which implys ==> G = 2 * (-x), X = V_uav

        Based on CBF theory, we must find an optimal u_star which satisfy the
        constraint
                        h_dot(x) >= -alpha(h(x))

        And to fulfill the form of quadratic function that CVXOPT solver required,
        we turn the constraint equation into
                        -h_dot(x) <= alpha(h(x))

        Finally, we could obtain the result that
                        P = identity matrix
                        Q = -V_uav
                        G = - (2 * (-x))
                        H = (-x)^2 - d^2
        """
        u = u.astype(np.double)

        self.Q[0, 0] = -u[0]
        self.Q[1, 0] = -u[1]
        self.Q[2, 0] = -u[2]
        yaw_rate = u[3]

        pcl = o3d.geometry.PointCloud()
        if pcl_data is None:
            pcl.points = o3d.utility.Vector3dVector(self.pcl)
        else:
            pcl.points = o3d.utility.Vector3dVector(pcl_data)
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

        h = np.square(-pcl).sum(axis=1) - self.safe_dis**2
        g = -(2 * (-pcl))

        if self.height_constraint:
            h = np.append(h, np.array([self.height - self.min_height,
                                       self.max_height - self.height]), axis=0)
            g = np.append(g, -1 * np.array([[0, 0,  1],
                                            [0, 0, -1]]), axis=0)

        h = np.append(h, np.array([-self.min_velocity,
                                    self.max_velocity,
                                   -self.min_velocity,
                                    self.max_velocity,
                                   -self.min_velocity,
                                    self.max_velocity]), axis=0)
        g = np.append(g, -1 * np.array([[ 1,  0,  0],
                                        [-1,  0,  0],
                                        [ 0,  1,  0],
                                        [ 0, -1,  0],
                                        [ 0,  0,  1],
                                        [ 0,  0, -1]]), axis=0)

        self.H = matrix(h.astype(np.double), tc='d')
        self.G = matrix(g.astype(np.double), tc='d')

        u_star = solvers.coneqp(self.P, self.Q, self.G, self.H)['x']

        return np.array([u_star[0], u_star[1], u_star[2], yaw_rate])

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        action = self.constraint_solver(action)

        return action

    def train(self, states, actions, rewards, next_states, dones):
        assert isinstance(states, torch.Tensor), "[NFWPO] states should be torch.Tensor type"
        assert isinstance(actions, torch.Tensor), "[NFWPO] actions should be torch.Tensor type"
        assert isinstance(rewards, torch.Tensor), "[NFWPO] rewards should be torch.Tensor type"
        assert isinstance(next_states, torch.Tensor), "[NFWPO] next_states should be torch.Tensor type"
        assert isinstance(dones, torch.Tensor), "[NFWPO] dones should be torch.Tensor type"
        self.total_iter += 1

        with torch.no_grad():
            # compute next target actions
            policy_noise = (torch.randn_like(actions) * self.noise_scale).clamp(-self.max_noise, self.max_noise)
            next_actions = (self.target_actor(next_states) + policy_noise).clamp(self.action_low, self.action_high)

            # compute target
            next_target_Q1 = self.target_critic_1(next_states, next_actions)
            next_target_Q2 = self.target_critic_2(next_states, next_actions)
            next_target_Q = torch.min(next_target_Q1, next_target_Q2)
            target_Q = rewards + self.discount * (1.0 - dones) * next_target_Q

        # calculate critic loss
        Q1 = self.critic_1(states, actions)
        Q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        self.critic_loss = critic_loss.detach()

        # back-propagation update critic networks
        self.optimizer_critic_1.zero_grad()
        self.optimizer_critic_2.zero_grad()
        critic_loss.backward()
        self.optimizer_critic_1.step()
        self.optimizer_critic_2.step()

        # delayed policy update
        if self.total_iter % self.update_freq == 0:
            start = time.time()
            predicted_actions = self.actor(states)
            projected_actions = predicted_actions.clone().detach().requires_grad_()
            G_list = []
            H_list = []
            for i in range(len(predicted_actions)):
                with torch.no_grad():
                    pcl = self.state_to_pcl(states[i])
                    a = self.constraint_solver(projected_actions[i].cpu().numpy(), pcl)
                    projected_actions[i][0] = a[0]
                    projected_actions[i][1] = a[1]
                    projected_actions[i][2] = a[2]
                    projected_actions[i][3] = a[3]
                    G_list.append(self.G)
                    H_list.append(self.H)

            Q = self.critic_1(states, projected_actions)
            gradients = torch.autograd.grad(Q, projected_actions, grad_outputs=torch.ones_like(Q))[0]
            gradients = gradients.cpu().numpy().astype(np.double)
            action_table = torch.zeros((len(predicted_actions), 4)).to(self.device)

            for i in range(len(predicted_actions)):
                P = matrix(np.zeros((3, 3)), tc='d')
                c = matrix([-gradients[i][0], -gradients[i][1], -gradients[i][2]], tc='d')
                u_star = solvers.coneqp(P, c, G_list[i], H_list[i])['x']

                with torch.no_grad():
                    action_table[i][0] = u_star[0]
                    action_table[i][1] = u_star[1]
                    action_table[i][2] = u_star[2]
                    action_table[i][3] = projected_actions[i][3]

                    action_table[i] = 0.05 * action_table[i] + 0.95 * projected_actions[i]

            actor_loss = F.mse_loss(predicted_actions, action_table)
            self.actor_loss = actor_loss.detach()

            # back-propagation update actor networks
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            # soft update
            self._soft_update(self.actor, self.target_actor, tau=self.tau)
            self._soft_update(self.critic_1, self.target_critic_1, tau=self.tau)
            self._soft_update(self.critic_2, self.target_critic_2, tau=self.tau)

            print(f"elapsed: {time.time()-start}")

    def save_model(self, filename):
        checkpoints = {"NFWPO_actor": self.actor.state_dict(),
                       "NFWPO_critic_1": self.critic_1.state_dict(),
                       "NFWPO_critic_2": self.critic_2.state_dict(),
                       "NFWPO_optimizer_actor": self.optimizer_actor.state_dict(),
                       "NFWPO_optimizer_critic_1": self.optimizer_critic_1.state_dict(),
                       "NFWPO_optimizer_critic_2": self.optimizer_critic_2.state_dict()}

        torch.save(checkpoints, filename)
        print("[NFWPO] Model saved successfully")

    def load_model(self, filename):
        checkpoints = torch.load(filename)

        self.actor.load_state_dict(checkpoints["NFWPO_actor"])
        self.critic_1.load_state_dict(checkpoints["NFWPO_critic_1"])
        self.critic_2.load_state_dict(checkpoints["NFWPO_critic_2"])
        self.optimizer_actor.load_state_dict(checkpoints["NFWPO_optimizer_actor"])
        self.optimizer_critic_1.load_state_dict(checkpoints["NFWPO_optimizer_critic_1"])
        self.optimizer_critic_2.load_state_dict(checkpoints["NFWPO_optimizer_critic_2"])

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        print("[NFWPO] Model loaded successfully")

    def _soft_update(self, current_net, target_net, tau=0.005):
        for current_param, target_param in zip(current_net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * current_param.data + (1 - tau) * target_param.data)

    def _initialize(self):
        rate = rospy.Rate(30)
        while self.pcl is None:
            rate.sleep()

if __name__ == "__main__":
    agent = NFWPO(1, 1, 1)
