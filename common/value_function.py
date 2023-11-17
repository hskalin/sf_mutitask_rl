import common.builder as builder
import torch
import torch.nn as nn


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

        self.Q = builder.create_linear_network(
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
        max_nheads=int(50),
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.max_nheads = max_nheads
        self.n_heads = n_heads
        self.layernorm = layernorm
        self.fuzzytiling = fuzzytiling

        self.model = builder.create_multihead_linear_model(
            input_dim=observation_dim + action_dim,
            output_dim=feature_dim * self.max_nheads,
            hidden_units=sizes,
            hidden_activation=activation,
            output_activation=None,
            layernorm=layernorm,
            fuzzytiling=fuzzytiling,
            initializer=initializer,
        )

    def forward(self, observations, actions):
        x = torch.cat([observations, actions], dim=1)
        x = self.model(x)
        x = x.view(-1, self.max_nheads, self.feature_dim)
        return x[:, : self.n_heads, :]

    def forward_head(self, observations, actions, head_idx):
        x = self.forward(observations, actions)
        return x[:, head_idx, :]

    def add_head(self, n_heads=1):
        self.n_heads += n_heads


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


if __name__ == "__main__":
    from torch.profiler import ProfilerActivity, profile, record_function

    obs_dim = 5
    featdim = 10
    act_dim = 2
    n_heads = 100

    layernorm = True
    fuzzytiling = True
    device = "cuda"

    times = 1111
    obs = torch.rand(1000, obs_dim).to(device)
    act = torch.rand(1000, act_dim).to(device)

    sfn = TwinnedMultiheadSFNetwork(
        observation_dim=obs_dim,
        feature_dim=featdim,
        action_dim=act_dim,
        n_heads=n_heads,
        layernorm=layernorm,
        fuzzytiling=fuzzytiling,
    ).to(device)
    sfn_scripted = torch.jit.script(sfn)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        use_cuda=True,
        with_stack=True,
    ) as prof1:
        with record_function("model_inference"):
            for _ in range(times):
                sfn_scripted(obs, act)

    print(prof1.key_averages().table(sort_by="cuda_time_total", row_limit=10))
