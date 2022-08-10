import rospy
import numpy as np
import networkx as nx
from pqdict import pqdict
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
from ..envs.px4_env import PX4Env
from .local_map_generator import LocalMapGenerator

class ForwardSpanningTreePlanner:
    def __init__(self, sensor_dis, map_precision, r_n=10.0, n_samples=500, visualize=True):
        self.local_map = None
        self.local_map_generator = LocalMapGenerator(sensor_dis, map_precision)

        self.nodes_pos = list()
        self.nodes_ckd_tree = None
        self.obstacles_ckd_tree = None
        self.fast_spanning_tree = nx.Graph()

        self.sensor_dis = sensor_dis
        self.map_precision = map_precision
        self.r_n = r_n
        self.n_samples = n_samples
        self.visualize = visualize
        self.map_resolution = np.floor(2 * sensor_dis / map_precision).astype(np.float64) + 1
        self.map_center_offset = np.floor(self.map_resolution / 2).astype(np.float64)

        if self.visualize:
            plt.ion()
            self.fig = plt.figure("Forward Spanning Tree Planner")

    def plan(self, goal):
        self.local_map = self.local_map_generator.generate()
        self.obstacles_ckd_tree = cKDTree(np.argwhere(self.local_map == 0))

        start, goal = self._info_preprocessing([0.0, 0.0], goal)

        self.nodes_pos = self.sample_free_space(self.n_samples)
        start_id = len(self.nodes_pos)
        goal_id = start_id + 1
        for i, node in zip([start_id, goal_id], [start, goal]):
            self.fast_spanning_tree.add_node(i)
            self.nodes_pos.append(node)
        self.nodes_ckd_tree = cKDTree(self.nodes_pos)

        V_unvisited = list(range(len(self.nodes_pos)))
        V_unvisited.remove(start_id)
        V_open = pqdict({start_id: 0})
        V_closed = list()
        z = start_id

        plannable = True
        while z != goal_id:
            N_z = self.fixed_distance_neighbors(z, self.r_n)
            X_near = list(set(N_z) & set(V_unvisited))
            for x in X_near:
                N_x = self.fixed_distance_neighbors(x, self.r_n)
                Y_near = list(set(N_x) & set(V_open))
                y_min = Y_near[np.argmin([V_open[y] for y in Y_near])]
                if self.check_collision_free(self.nodes_pos[y_min], self.nodes_pos[x]):
                    self.fast_spanning_tree.add_edge(y_min, x)
                    cost = V_open[y_min] + np.linalg.norm(self.nodes_pos[y_min] - self.nodes_pos[x])
                    if x in V_open:
                        V_open.updateitem(x, cost)
                    else:
                        V_open.additem(x, cost)
                    V_unvisited.remove(x)
            V_open.pop(z)
            V_closed.append(z)
            if len(V_open) == 0:
                plannable = False
                break
            z = V_open.top()

        if plannable:
            path = np.vstack([self.nodes_pos[x] for x in nx.shortest_path(self.fast_spanning_tree, start_id, z)])
        else:
            path = np.vstack([self.nodes_pos[start_id], self.nodes_pos[goal_id]])

        if self.visualize:
            plt.clf()
            plt.imshow(self.local_map, cmap="gray")
            nx.draw(self.fast_spanning_tree, [x[::-1] for x in self.nodes_pos], node_size=1, alpha=.5)
            if plannable:
                plt.plot(path[:, 1], path[:, 0], 'r-', lw=2)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        path = self._info_postprocessing(path)
        return path, plannable

    def fixed_distance_neighbors(self, z, r_n):
        neighbors = self.nodes_ckd_tree.query_ball_point(self.nodes_pos[z], r_n)
        return neighbors

    def sample_free_space(self, n):
        self.fast_spanning_tree.clear()
        nodes_pos = list()
        i = 0
        while len(nodes_pos) < n:
            node = np.random.uniform(0, self.local_map.shape)
            if self.check_collision_free(node):
                self.fast_spanning_tree.add_node(i)
                nodes_pos.append(node)
                i += 1
        return nodes_pos

    def check_collision_free(self, src, dst=None):
        if (dst is None) or np.all(src == dst):
            return self.obstacles_ckd_tree.query(src)[0] > self.r_n/2

        dx, dy = dst[0] - src[0], dst[1] - src[1]
        angle = np.arctan2(dy, dx)
        dis = np.hypot(dx, dy)
        steps = np.arange(0, dis, self.r_n/10).reshape(-1, 1)
        pts = src + steps * np.array([np.cos(angle), np.sin(angle)])
        pts = np.vstack((pts, dst))
        return bool(self.obstacles_ckd_tree.query(pts)[0].min() > self.r_n/2)

    def _info_preprocessing(self, start, goal):
        start, goal = np.asarray(start), np.asarray(goal)
        start += self.map_center_offset
        goal_dis = np.linalg.norm(goal)
        if goal_dis > self.sensor_dis:
            goal = (goal / goal_dis * self.sensor_dis).astype(np.int32)
        goal = (goal / self.map_precision * -1).astype(np.int32) + self.map_center_offset

        while not self.check_collision_free(goal):
            vector = self.map_center_offset - goal
            vector_norm = np.linalg.norm(vector)
            if vector_norm < self.r_n:
                break
            goal += vector / vector_norm * self.r_n
        assert self.check_collision_free(goal), "[ERROR] Goal is trapped in obstacles"
        return start, goal

    def _info_postprocessing(self, path):
        path = (path - self.map_center_offset) * -1 * self.map_precision
        dist = np.linalg.norm(path, axis=1).reshape(-1, 1)
        ang = np.arctan2(path[:, 1], path[:, 0]).reshape(-1, 1)
        goal_info = np.concatenate([dist, ang], axis=1)
        return goal_info

if __name__ == "__main__":
    rospy.init_node("motion_planning_node")
    SENSOR_WIDTH = 6.0
    MAP_PRECISION = 0.1 # m
    R_N = 10.0
    N_SAMPLES = 500
    VISUALIZE = True

    steps = 300
    env = PX4Env()
    state = env.reset()
    fst = ForwardSpanningTreePlanner(SENSOR_WIDTH, MAP_PRECISION, r_n=R_N, n_samples=N_SAMPLES, visualize=VISUALIZE)

    for i in range(300):
        goal_x = state[-6] * np.cos(state[-5])
        goal_y = state[-6] * np.sin(state[-5])

        plan, plannable = fst.plan([goal_x, goal_y])

        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        if done:
            break
        state = next_state
