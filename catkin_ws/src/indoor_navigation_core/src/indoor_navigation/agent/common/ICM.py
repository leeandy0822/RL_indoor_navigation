import torch
import torch.nn as nn

class ForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hiddens=None):
        super(ForwardModel, self).__init__()

        if hiddens is None:
            hiddens = [64, 32]

        assert isinstance(hiddens, list), "[ForwardModel] hidden layers argument must be a list"

        layers = []
        hidden_ = state_dim + action_dim
        for hidden in hiddens:
            layers.append(nn.Linear(hidden_, hidden))
            layers.append(nn.ELU())
            hidden_ = hidden
        layers.append(nn.Linear(hidden_, state_dim))
        self.net = nn.Sequential(*layers)

        self._init_weights(self.net)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        next_state_hat = self.net(sa)

        return next_state_hat

    def _init_weights(self, net):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

class InverseModel(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hiddens=None):
        super(InverseModel, self).__init__()
        self.max_action = max_action

        if hiddens is None:
            hiddens = [32]

        assert isinstance(hiddens, list), "[InverseModel] hidden layers argument must be a list"

        layers = []
        hidden_ = 2 * state_dim
        for hidden in hiddens:
            layers.append(nn.Linear(hidden_, hidden))
            layers.append(nn.ELU())
            hidden_ = hidden
        layers.append(nn.Linear(hidden_, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

        self._init_weights(self.net)

    def forward(self, state, next_state):
        ss_ = torch.cat([state, next_state], dim=-1)
        a_hat = self.net(ss_)

        return self.max_action * a_hat

    def _init_weights(self, net):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

class ICM:
    def __init__(self, state_dim, action_dim, max_action, to_cuda=True, **kwargs):
        self.lambda_i = kwargs.pop("lambda_i", 1)
        self.eta = kwargs.pop("eta", 1)
        self.beta = kwargs.pop("beta", 0.2)

        if to_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.forward_model = ForwardModel(state_dim, action_dim, hiddens=[64, 32]).to(self.device)
        self.inverse_model = InverseModel(state_dim, action_dim, max_action, hiddens=[32]).to(self.device)

        self.icm_fwd_optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=3e-4)
        self.icm_inv_optimizer = torch.optim.Adam(self.inverse_model.parameters(), lr=3e-4)

    def train(self, states, actions, next_states):
        assert isinstance(states, torch.Tensor), "[ICM] states should be torch.Tensor type"
        assert isinstance(actions, torch.Tensor), "[ICM] actions should be torch.Tensor type"
        assert isinstance(next_states, torch.Tensor), "[ICM] next_states should be torch.Tensor type"

        next_states_hat = self.forward_model(states, actions)
        actions_hat = self.inverse_model(states, next_states)

        forward_losses = 0.5 * torch.square(next_states - next_states_hat).mean(dim=-1, keepdim=True)
        inverse_losses = 0.5 * torch.square(actions - actions_hat).mean(dim=-1, keepdim=True)
        L_f = forward_losses.mean().detach()
        L_i = inverse_losses.mean().detach()
        intrinsic_rewards = self.eta * (forward_losses.detach())

        icm_loss = (1 - self.beta) * inverse_losses.mean() + self.beta * forward_losses.mean()
        self.icm_fwd_optimizer.zero_grad()
        self.icm_inv_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_fwd_optimizer.step()
        self.icm_inv_optimizer.step()

        return L_f, L_i, self.lambda_i * intrinsic_rewards

    def save_model(self, filename):
        checkpoints = {"ICM_forward_model": self.forward_model.state_dict(),
                       "ICM_inverse_model": self.inverse_model.state_dict(),
                       "ICM_optimizer_forward_model": self.icm_fwd_optimizer.state_dict(),
                       "ICM_optimizer_inverse_model": self.icm_inv_optimizer.state_dict()}

        torch.save(checkpoints, filename)
        print("[ICM] Model saved successfully")

    def load_model(self, filename):
        checkpoints = torch.load(filename)

        self.forward_model.load_state_dict(checkpoints["ICM_forward_model"])
        self.inverse_model.load_state_dict(checkpoints["ICM_inverse_model"])
        self.icm_fwd_optimizer.load_state_dict(checkpoints["ICM_optimizer_forward_model"])
        self.icm_inv_optimizer.load_state_dict(checkpoints["ICM_optimizer_inverse_model"])

        print("[ICM] Model loaded successfully")

if __name__ == "__main__":
    fm = ForwardModel(3, 2)
    print(fm.net)
    im = InverseModel(3, 2, 1)
    print(im.net)
