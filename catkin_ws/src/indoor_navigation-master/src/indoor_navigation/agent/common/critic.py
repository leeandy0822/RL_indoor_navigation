import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.layers = nn.Sequential(
                        nn.Linear(state_dim + action_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 1)
                      )

        self._init_weights()

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q = self.layers(sa)

        return q

    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)

if __name__ == "__main__":
    critic = Critic(2, 2, 1)
    print(critic)
