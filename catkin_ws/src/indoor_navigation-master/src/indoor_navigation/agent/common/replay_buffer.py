import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6), to_cuda=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size

        self.states = np.zeros((max_size, state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.rewards = np.zeros((max_size, 1))
        self.next_states = np.zeros((max_size, state_dim))
        self.dones = np.zeros((max_size, 1))

        self.cur = 0
        self.size = 0
        if to_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def add(self, state, action, reward, next_state, done):
        self.states[self.cur] = state
        self.actions[self.cur] = action
        self.rewards[self.cur] = reward
        self.next_states[self.cur] = next_state
        self.dones[self.cur] = done

        self.cur = (self.cur + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        sample_states = torch.FloatTensor(self.states[idx]).to(self.device)
        sample_actions = torch.FloatTensor(self.actions[idx]).to(self.device)
        sample_rewards = torch.FloatTensor(self.rewards[idx]).to(self.device)
        sample_next_states = torch.FloatTensor(self.next_states[idx]).to(self.device)
        sample_dones = torch.FloatTensor(self.dones[idx]).to(self.device)

        return sample_states, sample_actions, sample_rewards, sample_next_states, sample_dones

    def save(self, filename):
        kwargs = {"state_dim": self.state_dim,
                  "action_dim": self.action_dim,
                  "max_size": self.max_size,
                  "states": self.states,
                  "actions": self.actions,
                  "rewards": self.rewards,
                  "next_states": self.next_states,
                  "dones": self.dones,
                  "cur": self.cur,
                  "size": self.size}

        np.savez(filename, **kwargs)
        print(f"[ReplayBuffer] {self.size} datas saved")

    def load(self, filename):
        data = np.load(filename)

        if self.state_dim != data["state_dim"] or self.action_dim != data["action_dim"]:
            print("[ReplayBuffer] Data Dimension Not Matched")
            exit(1)

        self.max_size = data["max_size"]
        self.states = data["states"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.next_states = data["next_states"]
        self.dones = data["dones"]
        self.cur = data["cur"]
        self.size = data["size"]

        print(f"[ReplayBuffer] {self.size} datas loaded")

if __name__ == "__main__":
    replay_buffer = ReplayBuffer(2, 2, 10, to_cuda=False)

    datas = np.arange(70).reshape(10, 7)
    for data in datas:
        replay_buffer.add(data[:2], data[2:4], data[4], data[5:], True)

    states, actions, rewards, next_states, dones = replay_buffer.sample(3)
    print(states)
    print(actions)
    print(rewards)
    print(next_states)
    print(dones)

    replay_buffer.save("/tmp/test.npz")

    another_replay_buffer = ReplayBuffer(2, 2, 7, to_cuda=False)
    another_replay_buffer.load("/tmp/test.npz")
