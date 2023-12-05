import isaacgym

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.optim import Adam
from env.wrapper.multiTask import multitaskenv_constructor

import hydra
import wandb
from omegaconf import DictConfig
from common.util import omegaconf_to_dict, print_dict, fix_wandb, update_dict


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0
    ):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        # 1x1 convolution for residual connection
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # residual connection
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i # dilated conv
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_channels, kernel_size=2, dropout=0):
        super().__init__()
        self.tcn = TemporalConvNet(in_dim, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], out_dim)

    def forward(self, x):
        x = self.tcn(x)
        x = self.linear(x[:, :, -1])
        return x


def _verify_tcn(device):
    print("device: ", device)
    print("validating TCN performance")

    # parameters
    obs_dim = 11
    out_dim = 11
    # I suppose these are hidden layer sizes? so the obs dim is reduced to this size
    num_channels = [5, 7, 5] 
    kernel_size = 3

    # data
    batch_size = 50
    n_sample = 1000
    sequence_length = 17

    x = torch.rand(n_sample, obs_dim, sequence_length, device=device)
    y = x[...,-1].clone()

    # init model
    tcnnet = TCN(
        in_dim=obs_dim,
        out_dim=out_dim,
        num_channels=num_channels,
        kernel_size=kernel_size,
    )
    tcnnet = tcnnet.to(device=device)

    # training
    print("training")
    optimizer = Adam(tcnnet.parameters(), lr=0.01, eps=1e-5)
    loss_fn = nn.MSELoss()
    
    # epochs
    for i in range(10):
        tcnnet.train()
        train_loss = 0.0
        for j in range(n_sample//batch_size):
            optimizer.zero_grad() # Reset gradients
            z = tcnnet(x[j*batch_size:(j+1)*batch_size]) # forwad pass
            J = loss_fn(z[...], y[j*batch_size:(j+1)*batch_size]) # calc loss
            J.backward() # backward pass
            optimizer.step() # update weights

            train_loss += (J.detach().item() - train_loss) / (j + 1)

        print(f"Epoch: {i}\t train loss: {train_loss}")


class Perception:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        self.channels = [40, 40]
        self.kernel_size = 5
        self.history = 10

        self.epochs = 20
        self.num_steps = 500

        self._init_env()

        self.tcn = TCN(
            in_dim=self.env.num_obs,
            out_dim=self.env.num_obs,
            num_channels=self.channels,
            kernel_size=self.kernel_size,
        )

        self.optimizer = Adam(self.tcn.parameters(), lr=0.005, eps=1e-5)
        self.loss_fn = nn.MSELoss()

        # obs buffer to store sequences
        self.obs = torch.zeros(
            (self.num_envs, self.env.num_obs, self.num_steps),
            device=self.device, dtype=torch.float
        )

    def _init_env(self):
        self.num_envs = self.cfg["env"]["num_envs"]
        self.device = self.cfg["rl_device"]
        self.env, _, _ = multitaskenv_constructor(self.cfg["env"], device=self.device)

    def train(self):
        self.tcn.to(self.device)
        next_obs = self.env.obs_buf

        for epoch in range(self.epochs):
            self.obs[...,0] = next_obs

            self.tcn.train()
            mseloss = 0.0

            for step in range(1, self.num_steps):
                
                # have to find a good way to get these
                #actions = torch.rand((self.num_envs, self.env.num_act), device=self.device)
                actions = torch.sin(torch.ones((
                    self.num_envs, self.env.num_act), device=self.device)*(step/100))

                self.env.step(actions)
                next_obs = self.env.obs_buf
                self.env.reset()

                self.obs[...,step] = next_obs
                prev_idx = max(0, step - self.history)

                self.optimizer.zero_grad()
                pred_obs = self.tcn(self.obs[..., prev_idx:step])
                J = self.loss_fn(pred_obs, next_obs)
                J.backward()
                self.optimizer.step()

                #print(J.detach().item())
                mseloss += (J.detach().item() - mseloss) / (step + 1)

            print(f"Epoch: {epoch}\t train loss: {mseloss}")
        



@hydra.main(config_name="config", config_path="../cfg")
def launch_rlg_hydra(cfg: DictConfig):
    cfg_dict = omegaconf_to_dict(cfg)

    if cfg_dict["wandb_log"]:
        wandb.init()
    else:
        wandb.init(mode="disabled")
        
    print(wandb.config, "\n\n")
    wandb_dict = fix_wandb(wandb.config)

    print_dict(wandb_dict)
    update_dict(cfg_dict, wandb_dict)

    cfg_dict["buffer"]["n_env"] = cfg_dict["env"]["num_envs"]
    print_dict(cfg_dict)

    torch.manual_seed(cfg_dict["seed"])

    percep = Perception(cfg_dict)
    percep.train()

    wandb.finish()


if __name__ == "__main__":
    launch_rlg_hydra()