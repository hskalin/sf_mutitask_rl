import torch
import torch.nn as nn
import util
import distribution
from torch.distributions import Normal
import torch.jit as jit
from functorch import vmap, combine_state_for_ensemble
import builder
import activation_fn

class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, obs):
        raise NotImplementedError


class GaussianPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5
    eps = 1e-6

    def __init__(
        self,
        observation_dim,
        action_dim,
        sizes=[32, 32],
        squash=True,
        activation="relu",
        layernorm=False,
        initializer="xavier_uniform",
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes
        self.squash = squash

        output_activation = "tanh" if squash else None
        self.model = builder.create_linear_network(
            input_dim=observation_dim,
            output_dim=action_dim * 2,
            hidden_units=sizes,
            hidden_activation=activation,
            output_activation=output_activation,
            layernorm=layernorm,
            initializer=initializer,
        )
        nn.init.xavier_uniform_(self.model[-2].weight, 1e-3)

    def forward(self, obs):
        means, log_stds = self._forward(obs)
        normals, xs, actions = self.get_distribution(means, log_stds)
        entropy = self.calc_entropy(normals, xs, actions)
        return actions, entropy, means

    def _forward(self, obs):
        x = self.model(obs)
        means, log_stds = self.calc_mean_std(x)
        return means, log_stds

    def calc_mean_std(self, x):
        means, log_stds = torch.chunk(x, 2, dim=-1)
        log_stds = torch.clamp(log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        return means, log_stds

    def get_distribution(self, means, log_stds):
        stds = log_stds.exp()
        normals = Normal(means, stds)
        xs = normals.rsample()
        actions = torch.tanh(xs)
        return normals, xs, actions

    def calc_entropy(self, normals, xs, actions, dim=1):
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + self.eps)
        entropy = -log_probs.sum(dim=dim, keepdim=True)
        return entropy

    def sample(self, obs):
        return self.forward(obs)


class MultiheadGaussianPolicy(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        action_dim,
        n_heads,
        sizes=[64, 64],
        squash=True,
        activation="relu",
        layernorm=False,
        fuzzytiling=False,
        initializer="xavier_uniform",
        device="cpu",
        max_nheads=int(100)
    ):
        super().__init__()
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        self.eps = 1e-6

        self.device = device
        self.max_nheads = max_nheads
        self.n_heads = int(n_heads)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes
        self.squash = squash
        self.tanh = nn.Tanh() if squash else None

        self.model = builder.create_multihead_linear_model(
            input_dim=observation_dim,
            output_dim=2 * self.action_dim * self.max_nheads,
            hidden_units=sizes,
            hidden_activation=activation,
            output_activation=None,
            layernorm=layernorm,
            fuzzytiling=fuzzytiling,
            initializer=initializer,
        )
        nn.init.xavier_uniform_(self.model[-1].weight, 1e-3)

    def forward(self, obs):
        """forward multi-head"""
        means, log_stds = self._forward(obs)  # # [N, H, A], # [N, H, A]
        normals, xs, actions = self.get_distribution(means, log_stds)
        entropies = self.calc_entropy(normals, xs, actions, dim=1)  # [N, H, 1]
        return actions, entropies, means  # [N, H, A], [N, H, 1], [N, H, A]

    def _forward(self, obs):
        x = self.model(obs)  # [N, 2*H*A]
        x = x.view([-1, self.max_nheads, 2*self.action_dim]) # [N, H, 2*A]
        x = x[:, :self.n_heads, :]
        means, log_stds = self.calc_mean_std(x)  # [N, H, A], [N, H, A]
        return means, log_stds

    def forward_head(self, obs, idx):
        """forward single head"""
        actions, entropies, means = self.forward(obs)  # [N, H, A], [N, H, 1], [N, H, A]
        return actions[:, idx, :], entropies, means[:, idx, :]

    def calc_mean_std(self, x):
        means, log_stds = torch.chunk(x, 2, dim=-1)  # [N, H*A], [N, H*A] <-- [N, H, 2*A]
        log_stds = torch.clamp(log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        return means, log_stds

    def get_distribution(self, means, log_stds):
        stds = log_stds.exp()
        normals = Normal(means, stds)
        xs = normals.rsample()
        actions = self.tanh(xs) if self.squash else xs
        return normals, xs, actions

    def calc_entropy(self, normals, xs, actions, dim:int=1):
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + self.eps)
        entropy = -log_probs.sum(dim=dim, keepdim=True)
        return entropy
    
    def add_head(self, n_heads=1):
        self.n_heads += n_heads


class DynamicMultiheadGaussianPolicy(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        action_dim,
        n_heads,
        sizes=[64, 64],
        squash=True,
        activation="relu",
        layernorm=False,
        fuzzytiling=False,
        initializer="xavier_uniform",
        device="cpu",
    ):
        super().__init__()
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        self.eps = 1e-6

        self.device = device
        self.init_nheads = n_heads
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes
        self.squash = squash
        self.tanh = nn.Tanh() if squash else None

        base_model, base_out_dim = builder.create_linear_base(
            model=[],
            units=observation_dim,
            hidden_units=sizes,
            hidden_activation=activation,
            layernorm=layernorm,
        )

        if fuzzytiling:
            base_model.pop()
            fta = activation_fn.FTA()
            base_model.append(fta)
            base_out_dim *= fta.nbins

        self.base_out_dim = base_out_dim
        self.base_model = nn.Sequential(*base_model).apply(
            builder.initialize_weights(builder.str_to_initializer[initializer])
        ).to(self.device)

        self.n_heads = 0
        self.heads = nn.ModuleList([])
        self.params = None
        self.add_head(n_heads)
    
    def add_head(self, n_heads=1):
        for _ in range(n_heads):
            head = nn.Linear(self.base_out_dim, 2 * self.action_dim).to(self.device)
            nn.init.xavier_uniform_(head.weight, 1e-3)

            self.heads.append(head)
            self.n_heads += 1
        
        self._ensemble()

    def _ensemble(self):
        if self.params is not None:
            self._update_heads_param()

        fmodel, self.params, bufs = combine_state_for_ensemble(self.heads)
        [p.requires_grad_().to(self.device) for p in self.params]

        self.heads_model = lambda x: (vmap(fmodel, in_dims=(0, 0, None)))(self.params, bufs, x)
        
    def _update_heads_param(self):
        for i in range(self.n_heads):
            self.heads[i].weight.data = self.params[0][i].data
            self.heads[i].bias.data = self.params[1][i].data

    def forward(self, obs):
        """forward multi-head"""
        means, log_stds = self._forward(obs)
        normals, xs, actions = self.get_distribution(means, log_stds)
        entropies = self.calc_entropy(normals, xs, actions, dim=2)  # [N, H, 1]
        return actions, entropies, means  # [N, H, A], [N, H, 1], [N, H, A]

    def _forward(self, obs):
        """forward hidden layers"""
        x = self.base_model(obs)
        x = self.heads_model(x) # [H, N, 2*A]
        x = x.view([-1, self.n_heads, 2 * self.action_dim]) # [N, H, 2*A]
        means, log_stds = self.calc_mean_std(x)  # [N, H, A], [N, H, A]
        return means, log_stds
    
    def forward_head(self, obs, idx):
        """forward single head"""
        x = self.base_model(obs)
        x = self.heads[idx](x)
        means, log_stds = self.calc_mean_std(x)  # [N, A]
        normals, xs, actions = self.get_distribution(means, log_stds)
        entropies = self.calc_entropy(normals, xs, actions, dim=1)  # [N, 1]
        return actions, entropies, means

    def calc_mean_std(self, x):
        means, log_stds = torch.chunk(x, 2, dim=-1) # [N, H, A], [N, H, A] <-- [N, H, 2A]
        log_stds = torch.clamp(log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        return means, log_stds 

    def get_distribution(self, means, log_stds):
        stds = log_stds.exp()
        normals = Normal(means, stds)
        xs = normals.rsample()
        actions = self.tanh(xs) if self.squash else xs
        return normals, xs, actions

    def calc_entropy(self, normals, xs, actions, dim:int=1):
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + self.eps)
        entropy = -log_probs.sum(dim=dim, keepdim=True)
        return entropy


class GaussianMixturePolicy(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        action_dim,
        sizes=[64, 64],
        n_gauss=10,
        reg=0.001,
        reparameterize=True,
    ) -> None:
        super().__init__()
        self.eps = 1e-2

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes
        self.n_gauss = n_gauss
        self.reg = reg
        self.reparameterize = reparameterize

        self.model = distribution.GaussianMixture(
            input_dim=self.observation_dim,
            output_dim=self.action_dim,
            hidden_layers_sizes=sizes,
            K=n_gauss,
            reg=reg,
            reparameterize=reparameterize,
        )

    def forward(self, obs):
        act, logp, mean = self.model(obs)
        act = torch.tanh(act)
        mean = torch.tanh(mean)
        logp -= self.squash_correction(act)
        entropy = -logp[:, None].sum(
            dim=1, keepdim=True
        ) 
        return act, entropy, mean

    def squash_correction(self, inp):
        return torch.sum(torch.log(1 - torch.tanh(inp) ** 2 + self.eps), 1)

    def reg_loss(self):
        return self.model.reg_loss_t


class StochasticPolicy(BaseNetwork):
    """Stochastic NN policy"""

    def __init__(
        self,
        observation_dim,
        action_dim,
        sizes=[64, 64],
        squash=True,
        layernorm=False,
        activation="relu",
        initializer="xavier_uniform",
        device = "cpu",
    ):
        super().__init__()
        self.device = device

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes
        self.squash = squash

        output_activation = "tanh" if squash else None
        self.model = builder.create_linear_network(
            input_dim=self.observation_dim + self.action_dim,
            output_dim=action_dim * 2,
            hidden_units=sizes,
            hidden_activation=activation,
            output_activation=output_activation,
            layernorm=layernorm,
            initializer=initializer,
        )
        nn.init.xavier_uniform_(self.model[-2].weight, 1e-3)

    def forward(self, x):
        return self.model(x)

    def sample(self, obs):
        acts = self.get_actions(obs)
        a = acts[:, 0, :]  # TODO: add some randomness to explorative action
        return a, 0, a  # TODO: compute entropy

    def get_action(self, obs):
        return self.get_actions(obs).squeeze(0)[0]

    def get_actions(self, obs, n_act=1):
        obs = util.check_obs(obs)
        n_obs = obs.shape[0]

        latent_shape = (n_act, self.action_dim)
        latents = torch.normal(0, 1, size=latent_shape).to(self.device)

        s, a = util.get_sa_pairs(obs, latents)
        raw_actions = self.forward(torch.cat([s, a], -1)).view(
            n_obs, n_act, self.action_dim
        )

        return raw_actions


if __name__ == "__main__":
    from torch.profiler import profile, record_function, ProfilerActivity

    obs_dim = 5
    act_dim = 2
    n_heads = 100

    layernorm=True
    fuzzytiling=True
    device = "cuda"

    times = 1111
    obs = torch.rand(1000, obs_dim).to(device)

    # policy1 = DynamicMultiheadGaussianPolicy(
    #     observation_dim=obs_dim,
    #     action_dim=act_dim,
    #     n_heads=n_heads,
    #     layernorm=layernorm,
    #     fuzzytiling=fuzzytiling,
    #     device=device,
    # )

    policy2 = MultiheadGaussianPolicy(
        observation_dim=obs_dim,
        action_dim=act_dim,
        n_heads=n_heads,
        layernorm=layernorm,
        fuzzytiling=fuzzytiling,
        device=device,
    ).to(device)

#     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, use_cuda=True, with_stack=True,
# ) as prof1:
#         with record_function("model_inference"):        
#             for _ in range(times):
#                 policy1(obs)

#     print(prof1.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, use_cuda=True, with_stack=True,
) as prof2:
        with record_function("model_inference"):        
            for _ in range(times):
                policy2(obs)

    print(prof2.key_averages().table(sort_by="cuda_time_total", row_limit=10))
