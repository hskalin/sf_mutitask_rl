from numpy import argmax
import torch
import torch.nn as nn
from .builder import create_linear_network, create_multihead_linear_model

class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class QNetwork(BaseNetwork):
    # https://github.com/TakuyaHiraoka/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning/blob/main/KUCodebase/code/model.py

    def __init__(
        self,
        observation_dim,
        action_dim,
        sizes=[64, 64],
        activation="selu",
        layernorm=False,
        droprate=0,
        initializer="xavier",
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes
        self.activation = activation
        self.layernorm = layernorm
        self.droprate = droprate
        self.initializer = initializer

        self.Q = create_linear_network(
            self.observation_dim + self.action_dim,
            1,
            hidden_units=self.sizes,
            initializer=self.initializer,
            hidden_activation=self.activation,
        )

        new_q_networks = []
        for i, mod in enumerate(self.Q._modules.values()):
            new_q_networks.append(mod)
            if ((i % 2) == 0) and (i < (len(list(self.Q._modules.values()))) - 1):
                if self.droprate > 0.0:
                    new_q_networks.append(nn.Dropout(p=droprate))  # dropout
                if self.layernorm:
                    new_q_networks.append(nn.LayerNorm(mod.out_features))  # layer norm
            i += 1
        self.Q = nn.Sequential(*new_q_networks)

    def forward(self, x):
        q = self.Q(x)
        return q


class TwinnedQNetwork(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        action_dim,
        sizes=[64, 64],
        activation="selu",
        layernorm=False,
        droprate=0,
        initializer="xavier",
    ):
        super().__init__()

        self.Q1 = QNetwork(
            observation_dim,
            action_dim,
            sizes,
            activation,
            layernorm,
            droprate,
            initializer,
        )
        self.Q2 = QNetwork(
            observation_dim,
            action_dim,
            sizes,
            activation,
            layernorm,
            droprate,
            initializer,
        )

    def forward(self, observations, actions):
        x = torch.cat([observations, actions], dim=1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2


class MultiheadSFNetwork(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        feature_dim,
        action_dim,
        n_heads,
        sizes=[64, 64],
        activation="relu",
        layernorm=False,
        fuzzytiling=False,
        initializer="xavier_uniform",
    ) -> None:
        super().__init__()

        self.observation_dim = observation_dim
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.n_heads = n_heads
        self.layernorm = layernorm
        self.fuzzytiling = fuzzytiling
        
        self.model = create_multihead_linear_model(
            input_dim=observation_dim + action_dim,
            output_dim=feature_dim * self.n_heads,
            hidden_units=sizes,
            hidden_activation=activation,
            output_activation=None,
            layernorm=layernorm,
            fuzzytiling=fuzzytiling,
            initializer=initializer,
        )


    def forward(self, observations, actions):
        if observations.dim() == 1 and actions.dim() == 1:
            x = torch.cat([observations, actions], dim=0)
        else:
            x = torch.cat([observations, actions], dim=1)
        x = self.model(x)
        return x.view(-1, self.n_heads, self.feature_dim)

    def forward_head(self, observations, actions, head_idx):
        x = self.forward(observations, actions)
        return x[:, head_idx, :]


class TwinnedMultiheadSFNetwork(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        feature_dim,
        action_dim,
        n_heads,
        sizes=[64, 64],
        activation="relu",
        layernorm=False,
        fuzzytiling=False,
        initializer="xavier_uniform",
    ):
        super().__init__()

        self.SF1 = MultiheadSFNetwork(
            observation_dim,
            feature_dim,
            action_dim,
            n_heads,
            sizes,
            activation,
            layernorm,
            fuzzytiling,
            initializer,
        )
        self.SF2 = MultiheadSFNetwork(
            observation_dim,
            feature_dim,
            action_dim,
            n_heads,
            sizes,
            activation,
            layernorm,
            fuzzytiling,
            initializer,
        )

    def forward(self, observations, actions):
        sf1 = self.SF1(observations, actions)
        sf2 = self.SF2(observations, actions)
        return sf1, sf2
