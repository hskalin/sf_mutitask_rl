import torch
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from tensordict import TensorDict


class VectorizedReplayBuffer:
    def __init__(
        self,
        obs_shape,
        action_shape,
        feature_shape,
        capacity,
        device,
        mini_batch_size=64,
        *args,
        **kwargs,
    ):
        """Create Vectorized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        See Also
        --------
        ReplayBuffer.__init__
        """

        self.device = device
        self.batch_size = mini_batch_size

        self.obses = torch.empty(
            (capacity, *obs_shape), dtype=torch.float32, device=self.device
        )
        self.features = torch.empty(
            (capacity, *feature_shape), dtype=torch.float32, device=self.device
        )
        self.next_obses = torch.empty(
            (capacity, *obs_shape), dtype=torch.float32, device=self.device
        )
        self.actions = torch.empty(
            (capacity, *action_shape), dtype=torch.float32, device=self.device
        )
        self.rewards = torch.empty(
            (capacity, 1), dtype=torch.float32, device=self.device
        )
        self.dones = torch.empty((capacity, 1), dtype=torch.bool, device=self.device)

        self.capacity = capacity
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.idx

    def add(self, obs, feature, action, reward, next_obs, done):
        num_observations = obs.shape[0]
        remaining_capacity = min(self.capacity - self.idx, num_observations)
        overflow = num_observations - remaining_capacity
        if remaining_capacity < num_observations:
            self.obses[0:overflow] = obs[-overflow:]
            self.features[0:overflow] = feature[-overflow:]
            self.actions[0:overflow] = action[-overflow:]
            self.rewards[0:overflow] = reward[-overflow:]
            self.next_obses[0:overflow] = next_obs[-overflow:]
            self.dones[0:overflow] = done[-overflow:]
            self.full = True
        self.obses[self.idx : self.idx + remaining_capacity] = obs[:remaining_capacity]
        self.features[self.idx : self.idx + remaining_capacity] = feature[
            :remaining_capacity
        ]
        self.actions[self.idx : self.idx + remaining_capacity] = action[
            :remaining_capacity
        ]
        self.rewards[self.idx : self.idx + remaining_capacity] = reward[
            :remaining_capacity
        ]
        self.next_obses[self.idx : self.idx + remaining_capacity] = next_obs[
            :remaining_capacity
        ]
        self.dones[self.idx : self.idx + remaining_capacity] = done[:remaining_capacity]

        self.idx = (self.idx + num_observations) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size=None):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obses: torch tensor
            batch of observations
        actions: torch tensor
            batch of actions executed given obs
        rewards: torch tensor
            rewards received as results of executing act_batch
        next_obses: torch tensor
            next set of observations seen after executing act_batch
        not_dones: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not
        not_dones_no_max: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not, specifically exlcuding maximum episode steps
        """
        if batch_size is None:
            batch_size = self.batch_size

        idxs = torch.randint(
            0,
            self.capacity if self.full else self.idx,
            (batch_size,),
            device=self.device,
        )
        obses = self.obses[idxs]
        features = self.features[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_obses = self.next_obses[idxs]
        dones = self.dones[idxs]

        return {
            "obs": obses,
            "feature": features,
            "action": actions,
            "reward": rewards,
            "next_obs": next_obses,
            "done": dones,
        }


class FrameStackedReplayBuffer(VectorizedReplayBuffer):
    def __init__(
        self,
        obs_shape,
        action_shape,
        feature_shape,
        capacity,
        n_env,
        stack_size,
        device,
        mini_batch_size=64,
    ):
        super().__init__(
            obs_shape=obs_shape,
            action_shape=action_shape,
            feature_shape=feature_shape,
            capacity=capacity,
            device=device,
            mini_batch_size=mini_batch_size,
        )
        self.n_env = n_env
        self.stack_size = stack_size

        self.stacked_obses = torch.empty(
            (capacity, *obs_shape, stack_size), dtype=torch.float32, device=self.device
        )

        self.ra = self.n_env * torch.arange(1, self.stack_size + 1)
        self.ra = self.ra.to(self.device)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        idxs = torch.randint(
            0,
            self.capacity if self.full else self.idx,
            (batch_size,),
            device=self.device,
        )
        obses = self.obses[idxs]  # [N, F]
        prev_obses = self.stack_obs(idxs)  # [N, F, S]
        features = self.features[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_obses = self.next_obses[idxs]
        dones = self.dones[idxs]

        return {
            "obs": obses,
            "prev_obses": prev_obses,
            "feature": features,
            "action": actions,
            "reward": rewards,
            "next_obs": next_obses,
            "done": dones,
        }

    def stack_obs(self, idx):
        # select idxes
        ra = self.ra.repeat((idx.shape[0], 1))
        ids = idx[:, None] - ra  # [N sample, N stack] <-- [N sample]

        # if full then go back
        if self.full:
            obs_stacked = self.obses[ids]  # [N sample, N stack, S]
        else:
            obs_stacked = self.obses[torch.where(ids < 0, -1, ids)]
            obs_stacked[ids < 0] = 0

        # correct by dones
        dop = self.dones[ids]  # [N sample, N stack, 1]
        m = torch.where(torch.cumsum(dop, 1) > 0, 0, 1)
        obs_stacked = m * obs_stacked  # [N sample, N stack, S]
        return obs_stacked


class VecPrioritizedReplayBuffer:
    def __init__(
        self,
        capacity,
        device,
        alpha=0.6,
        beta=0.4,
        mini_batch_size=64,
        *args,
        **kwargs,
    ):
        self.device = device
        self.rb = TensorDictPrioritizedReplayBuffer(
            alpha=alpha,
            beta=beta,
            storage=LazyTensorStorage(capacity, device=self.device),
            batch_size=mini_batch_size,
        )

    def add(self, obs, feature, action, reward, next_obs, done):
        data = TensorDict(
            {
                "obs": obs,
                "feature": feature,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
            },
            obs.shape[0],
        )
        self.rb.extend(data)

    def sample(self, ndata=None):
        if ndata is not None:
            d = self.rb.sample(ndata)
        else:
            d = self.rb.sample()
        return d

    def update_tensordict_priority(self, sample):
        self.rb.update_tensordict_priority(sample)

    def __len__(self):
        return len(self.rb)


if __name__ == "__main__":
    capacity = 10
    device = "cuda"
    data_size = 10
    stack_size = 5

    n_env = 1
    obs_dim = 5
    feat_dim = 3
    act_dim = 2
    rew_dim = 1
    done_dim = 1

    obs = torch.rand(data_size, obs_dim)
    feature = torch.rand(data_size, feat_dim)
    action = torch.rand(data_size, act_dim)
    reward = torch.rand(data_size, rew_dim)
    next_obs = torch.rand(data_size, obs_dim)
    done = torch.randint(0, 2, (data_size, done_dim))

    buf = FrameStackedReplayBuffer(
        obs_shape=(obs_dim,),
        action_shape=(act_dim,),
        feature_shape=(feat_dim,),
        capacity=capacity,
        n_env=n_env,
        stack_size=stack_size,
        device=device,
    )

    buf.add(obs, feature, action, reward, next_obs, done)
    print(len(buf))

    sample = buf.sample(1)
    print("buf.obs", buf.obses)
    print("buf.dones", buf.dones)
    print("obs", sample["obs"])
    print("stacked_obses", sample["stacked_obses"])
