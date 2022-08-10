import os
import json
import rospy
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from indoor_navigation.envs.px4_env import PX4Env
from indoor_navigation.agent.TD3 import TD3
from indoor_navigation.agent.common.replay_buffer import ReplayBuffer
from indoor_navigation.obstacle_avoidance.cbf import ConstraintGenerator
from indoor_navigation.motion_planning.fst import ForwardSpanningTreePlanner

class EnergyConsumptionCalculator:
    def __init__(self, W=39.2, A=0.18, Rho=1.225, C_D0=0.08, dt=0.2):
        self.W = W
        self.A = A
        self.Rho = Rho
        self.C_D0 = C_D0
        self.dt = dt

        self.v_h = np.sqrt(self.W / (2 * self.Rho * self.A))
        self.C1 = np.square(self.W) / (np.sqrt(2) * self.Rho * self.A)
        self.C2 = self.C_D0 * self.Rho * self.A / 8

        self._cumulated_energy = 0.0

    def reset(self):
        self._cumulated_energy = 0.0

    def add(self, v_x, v_y, v_z):
        v_xy_norm = np.linalg.norm(np.array([v_x, v_y]))

        P_level = self.C1 / np.sqrt(v_xy_norm**2 + np.sqrt(v_xy_norm**4 + 4 * self.v_h**4))
        P_vertical = self.W * v_z
        P_drag = self.C2 * v_xy_norm**3
        P_UAV = P_level + P_vertical + P_drag

        self._cumulated_energy += P_UAV * self.dt

    @property
    def power_consumption(self):
        return self._cumulated_energy

if __name__ == "__main__":
    rospy.init_node("test_node")
    parser = argparse.ArgumentParser()
    ###################   Environment Argument   ###################
    parser.add_argument("--env", default="PX4Env", type=str, choices=["PX4Env"], help="Choose a training environment")
    parser.add_argument("--reward_mode", default="dense", type=str, choices=["dense", "sparse"], help="Choose a reward mode")
    parser.add_argument("--episode_num", default=10000, type=int, help="Specify total training episodes")
    parser.add_argument("--max_timesteps", default=300, type=int, help="Sepcify max timesteps for each episode")
    parser.add_argument("--constraint", default="height", type=str, choices=["none", "height", "full"], help="Choose a constraint scenario")
    parser.add_argument("--planner", default="none", type=str, choices=["none", "fst"], help="Choose a motion planner")
    ###################   Policy Argument   ###################
    parser.add_argument("--policy", default="TD3", type=str, choices=["TD3"], help="Choose a policy to interact with environment")
    ###################   Testing Argument   ###################
    parser.add_argument("--name", default="", type=str, help="The name of the simulation")
    parser.add_argument("--seed", type=int, help="Set random seed")
    parser.add_argument("--load_model", default="", type=str, help="Load a model")
    parser.add_argument("--save_buffer", action="store_true", help="Save the replay buffer or not")
    parser.add_argument("--visualize", action="store_true", help="Visualize the comparison between desired action and constrained action")
    parser.add_argument("--cal_violation", action="store_true", help="Calculate the number of constraint violations every time steps")
    args = parser.parse_args()

    # select environment
    if args.env == "PX4Env":
        env = PX4Env(reward_mode=args.reward_mode)
        vehicle_type = rospy.get_param("/px4/vehicle_type")
    else:
        exit(f"[Error] \033[91mEnvironment {args.env} not supported\033[0m")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = min(env.action_space.high)
    min_action = max(env.action_space.low)

    # select agent
    if args.policy == "TD3":
        agent = TD3(state_dim, action_dim, max_action)
    else:
        exit(f"[Error] \033[91mPolicy {args.policy} not supported\033[0m")

    if args.load_model != "":
        agent.load_model(args.load_model)

    # create replay buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Set random seeds
    if args.seed != None:
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # record simulation data
    if not os.path.exists("./simulation/test"):
        os.makedirs("./simulation/test")

    if args.name == "":
        simulation_name = f"{args.env}_{args.policy}"
    else:
        simulation_name = args.name

    save_path = f"./simulation/test/{simulation_name}"
    if os.path.exists(save_path):
        save_path_serial = 1
        while os.path.exists(f"{save_path}_{save_path_serial}"):
            save_path_serial += 1
        save_path += f"_{save_path_serial}"
        os.makedirs(save_path)

    if not os.path.exists(save_path + "/logs"):
        os.makedirs(save_path + "/logs")

    with open(save_path + "/args_records.txt", "w") as f:
        args_dict = vars(args)
        json.dump({"args": args_dict}, sort_keys=True, indent=4, fp=f)

    writer = SummaryWriter(save_path + "/logs")

    # create constraint generator
    if args.constraint != "none":
        cg = ConstraintGenerator(height_constraint=True, min_height=0.5, max_height=2.0)

        if args.visualize:
            labels = ['Vx', 'Vy', 'Vz', 'Vw']
            x = np.arange(len(labels))
            width = 0.35

            plt.ion()
            figure, ax = plt.subplots(num="Obstacle Avoidance")
            desired = ax.bar(x - width/2, np.zeros(4), width, label='desired velocity', color='r')
            constrained = ax.bar(x + width/2, np.zeros(4), width, label='constrained velocity', color='g')
            ax.set_ylim([-0.3, 0.3])
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Velocity")
            ax.set_ylabel("m/s")
            ax.set_title("CBF Control")
            ax.legend()

            for i in range(4):
                desired.patches[i].set_height(0.0)
                constrained.patches[i].set_height(0.0)
            figure.canvas.draw()
            figure.canvas.flush_events()

    # create motion planner
    if args.planner != "none":
        planner = ForwardSpanningTreePlanner(6.0, 0.1, r_n=15.0, n_samples=500, visualize=True)

    # calculate constraint violations
    if args.cal_violation:
        cumulative_count = 0
        count_data = []
        epsilon = 0.1
        count_fig = plt.figure("cumulative constraint violation norm")

    # start testing
    try:
        global_steps = 0
        for ep in range(args.episode_num):
            if args.visualize:
                for i in range(4):
                    desired.patches[i].set_height(0.0)
                    constrained.patches[i].set_height(0.0)
                figure.canvas.draw()
                figure.canvas.flush_events()

            print(f"Episode: {ep+1}")
            episode_reward = 0.0
            state = env.reset()
            if vehicle_type == "iris_rplidar":
                state[:-6] /= 6.0
            elif vehicle_type == "iris_depth_camera":
                state[:-6] /= 255.0

            for t in range(args.max_timesteps):
                if args.planner != "none":
                    goal_x = state[-6] * np.cos(state[-5])
                    goal_y = state[-6] * np.sin(state[-5])
                    goal_info, plannable = planner.plan([goal_x, goal_y])
                    if plannable:
                        state[-6:-4] = goal_info[1]
                    else:
                        state[-6:-4] = goal_info[0]
                action = agent.select_action(state)

                if args.constraint == "none":
                    constrained_action = action
                elif args.constraint == "height":
                    constrained_action = cg.height_constraint_solver(action)
                elif args.constraint == "full":
                    constrained_action = cg.constraint_solver(action)

                if args.cal_violation:
                    norm = np.linalg.norm((constrained_action - action)[:2])
                    if norm > epsilon:
                        cumulative_count += 1
                    count_data.append(cumulative_count)

                next_state, reward, done, _ = env.step(constrained_action)
                if vehicle_type == "iris_rplidar":
                    next_state[:-6] /= 6.0
                elif vehicle_type == "iris_depth_camera":
                    next_state[:-6] /= 255.0
                episode_reward += reward

                replay_buffer.add(state, constrained_action, reward, next_state, done)
                writer.add_scalar("Action/Velocity_X", action[0], global_steps+1)
                writer.add_scalar("Action/Velocity_Y", action[1], global_steps+1)
                writer.add_scalar("Action/Velocity_Z", action[2], global_steps+1)
                writer.add_scalar("Action/Velocity_W", action[3], global_steps+1)
                global_steps += 1

                if done:
                    break
                state = next_state

                if args.visualize:
                    for i in range(4):
                        desired.patches[i].set_height(action[i])
                        constrained.patches[i].set_height(constrained_action[i])
                    figure.canvas.draw()
                    figure.canvas.flush_events()

            print(f"[INFO] Episode Reward: {episode_reward}, Data: {replay_buffer.size}")
            writer.add_scalar("Reward/Episode_Reward", episode_reward, ep+1)
            writer.add_scalar("Performance/Reach", env.reach, ep+1)
            writer.add_scalar("Performance/Collision", env.collision, ep+1)

            print()

    except Exception as e:
        print("\033[93m" + str(e) + "\033[0m")

    finally:
        if args.save_buffer:
            replay_buffer.save(save_path + "/replay_buffer.npz")

        if args.cal_violation:
            plt.plot(count_data)
            count_fig.savefig(save_path + "/constraint_violation.png")
