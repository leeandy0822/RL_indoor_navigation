import copy
import torch
import torch.nn.functional as F
from .common.actor import Actor
from .common.critic import Critic

class TD3:
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

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        return action

    def train(self, states, actions, rewards, next_states, dones):
        assert isinstance(states, torch.Tensor), "[TD3] states should be torch.Tensor type"
        assert isinstance(actions, torch.Tensor), "[TD3] actions should be torch.Tensor type"
        assert isinstance(rewards, torch.Tensor), "[TD3] rewards should be torch.Tensor type"
        assert isinstance(next_states, torch.Tensor), "[TD3] next_states should be torch.Tensor type"
        assert isinstance(dones, torch.Tensor), "[TD3] dones should be torch.Tensor type"
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
            actor_loss = -self.critic_1(states, self.actor(states)).mean()
            self.actor_loss = actor_loss.detach()

            # back-propagation update actor networks
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            # soft update
            self._soft_update(self.actor, self.target_actor, tau=self.tau)
            self._soft_update(self.critic_1, self.target_critic_1, tau=self.tau)
            self._soft_update(self.critic_2, self.target_critic_2, tau=self.tau)

    def save_model(self, filename):
        checkpoints = {"TD3_actor": self.actor.state_dict(),
                       "TD3_critic_1": self.critic_1.state_dict(),
                       "TD3_critic_2": self.critic_2.state_dict(),
                       "TD3_optimizer_actor": self.optimizer_actor.state_dict(),
                       "TD3_optimizer_critic_1": self.optimizer_critic_1.state_dict(),
                       "TD3_optimizer_critic_2": self.optimizer_critic_2.state_dict()}

        torch.save(checkpoints, filename)
        print("[TD3] Model saved successfully")

    def load_model(self, filename):
        checkpoints = torch.load(filename)

        self.actor.load_state_dict(checkpoints["TD3_actor"])
        self.critic_1.load_state_dict(checkpoints["TD3_critic_1"])
        self.critic_2.load_state_dict(checkpoints["TD3_critic_2"])
        self.optimizer_actor.load_state_dict(checkpoints["TD3_optimizer_actor"])
        self.optimizer_critic_1.load_state_dict(checkpoints["TD3_optimizer_critic_1"])
        self.optimizer_critic_2.load_state_dict(checkpoints["TD3_optimizer_critic_2"])

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        print("[TD3] Model loaded successfully")

    def _soft_update(self, current_net, target_net, tau=0.005):
        for current_param, target_param in zip(current_net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * current_param.data + (1 - tau) * target_param.data)

if __name__ == "__main__":
    agent = TD3(1, 1, 1)
