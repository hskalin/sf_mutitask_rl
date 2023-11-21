import copy
import datetime
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from common.util import AverageMeter, check_act, check_obs, dump_cfg, np2ts
from common.vec_buffer import VectorizedReplayBuffer, VecPrioritizedReplayBuffer
from env.wrapper.multiTask import multitaskenv_constructor

import wandb

warnings.simplefilter("once", UserWarning)
exp_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


class AbstractAgent(ABC):
    @abstractmethod
    def act(self, s):
        pass

    @abstractmethod
    def step(self):
        pass


class IsaacAgent(AbstractAgent):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.env_cfg = cfg["env"]
        self.agent_cfg = cfg["agent"]
        self.buffer_cfg = cfg["buffer"]
        self.device = cfg["rl_device"]

        self.env, self.feature, self.task = multitaskenv_constructor(
            env_cfg=self.env_cfg, device=self.device
        )
        assert self.feature.dim == self.task.dim, "feature and task dimension mismatch"

        self.n_env = self.env_cfg["num_envs"]
        self.env_max_steps = self.env_cfg["max_episode_length"]  # eval steps
        self.episode_max_step = self.env_cfg["episode_max_step"]
        self.log_interval = self.env_cfg["log_interval"]
        self.total_episodes = int(self.env_cfg["total_episodes"])
        self.total_timesteps = self.n_env * self.episode_max_step * self.total_episodes

        self.eval = self.env_cfg["eval"]
        self.eval_interval = self.env_cfg["eval_interval"]
        self.eval_episodes = self.env_cfg["eval_episodes"]
        self.save_model = self.env_cfg["save_model"]

        self.observation_dim = self.env.num_obs
        self.feature_dim = self.feature.dim
        self.action_dim = self.env.num_act
        self.observation_shape = [self.observation_dim]
        self.feature_shape = [self.feature_dim]
        self.action_shape = [self.action_dim]

        if self.buffer_cfg["prioritized_replay"]:
            self.replay_buffer = VecPrioritizedReplayBuffer(
                device=self.device,
                **self.buffer_cfg,
            )
        else:
            self.replay_buffer = VectorizedReplayBuffer(
                self.observation_shape,
                self.action_shape,
                self.feature_shape,
                device=self.device,
                **self.buffer_cfg,
            )
        self.mini_batch_size = int(self.buffer_cfg["mini_batch_size"])
        self.min_n_experience = int(self.buffer_cfg["min_n_experience"])

        self.gamma = int(self.agent_cfg["gamma"])
        self.updates_per_step = int(self.agent_cfg["updates_per_step"])
        self.reward_scale = int(self.agent_cfg["reward_scale"])

        if self.save_model:
            log_dir = (
                self.agent_cfg["name"]
                + "/"
                + self.env_cfg["env_name"]
                + "/"
                + exp_date
                + "/"
            )
            self.log_path = self.env_cfg["log_path"] + log_dir
            Path(self.log_path).mkdir(parents=True, exist_ok=True)
            dump_cfg(self.log_path + "cfg", cfg)

        self.steps = 0
        self.episodes = 0

        self.games_to_track = 100
        self.game_rewards = AverageMeter(1, self.games_to_track).to("cuda:0")
        self.game_lengths = AverageMeter(1, self.games_to_track).to("cuda:0")
        self.avgStepRew = AverageMeter(1, 20).to(self.device)

    def run(self):
        while True:
            self.train_episode()

            if self.eval and (self.episodes % self.eval_interval == 0):
                self.evaluate()
                if self.save_model:
                    self.save_torch_model()

            if self.steps > self.total_timesteps:
                break

    def train_episode(self, gui_app=None, gui_rew=None):
        self.episodes += 1
        episode_r = episode_steps = 0
        done = False

        print("episode = ", self.episodes)
        self.task.rand_task(self.episodes)

        s = self.reset_env()
        for _ in range(self.episode_max_step):
            # episodeRet = self.env.return_buf.clone()
            episodeLen = self.env.progress_buf.clone()

            s_next, r, done = self.step(episode_steps, s)

            s = s_next
            self.steps += self.n_env
            episode_steps += 1
            episode_r += r

            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.size()[0]:
                self.game_rewards.update(episode_r[done_ids])
                self.game_lengths.update(episodeLen[done_ids])

            # call gui update loop
            if gui_app:
                gui_app.update_idletasks()
                gui_app.update()
                self.avgStepRew.update(r)
                gui_rew.set(self.avgStepRew.get_mean())

            if episode_steps >= self.episode_max_step:
                break

        wandb.log({"reward/train": self.game_rewards.get_mean()})
        wandb.log({"length/train": self.game_lengths.get_mean()})

        return episode_r, episode_steps

    def step(self, episode_steps, s):
        a = self.act(s, self.task.Train)
        assert not torch.isinf(a).any(), "detect action infinity"

        self.env.step(a)
        done = self.env.reset_buf.clone()
        s_next = self.env.obs_buf.clone()
        self.env.reset()

        r = self.calc_reward(s_next, self.task.Train.W)
        assert not torch.isinf(r).any(), "detect reward infinity"

        masked_done = False if episode_steps >= self.episode_max_step else done
        self.save_to_buffer(s, a, r, s_next, done, masked_done)

        if self.is_update():
            for _ in range(self.updates_per_step):
                self.learn()

        return s_next, r, done

    def is_update(self):
        return (
            len(self.replay_buffer) > self.mini_batch_size
            and self.steps >= self.min_n_experience
        )

    def reset_env(self):
        s = self.env.obs_buf.clone()
        if s is None:
            s = torch.zeros((self.n_env, self.env.num_obs))

        return s

    def save_to_buffer(self, s, a, r, s_next, done, masked_done):
        f = self.feature.extract(s)
        r = r[:, None] * self.reward_scale
        done = done[:, None]
        masked_done = masked_done[:, None]

        self.replay_buffer.add(s, f, a, r, s_next, masked_done)

    def evaluate(self):
        episodes = int(self.eval_episodes)
        if episodes == 0:
            return

        print(
            f"===== evaluate at episode: {self.episodes} for {self.env_max_steps} steps ===="
        )

        returns = torch.zeros((episodes,), dtype=torch.float32)
        for i in range(episodes):
            episode_r = 0.0

            s = self.reset_env()
            for _ in range(self.env_max_steps):
                a = self.act(s, self.task.Eval, "exploit")
                self.env.step(a)
                s_next = self.env.obs_buf.clone()
                self.env.reset()

                r = self.calc_reward(s_next, self.task.Eval.W)

                s = s_next
                episode_r += r

            returns[i] = torch.mean(episode_r).item()

        print(f"===== finish evaluate ====")
        wandb.log({"reward/eval": torch.mean(returns).item()})

    def act(self, s, task, mode="explore"):
        w = copy.copy(np2ts(task.W))
        s = check_obs(s, self.observation_dim)

        with torch.no_grad():
            if (self.steps <= self.min_n_experience) and mode == "explore":
                a = 2 * torch.rand((self.n_env, self.env.num_act), device="cuda:0") - 1

            if mode == "explore":
                a = self.explore(s, w)
            elif mode == "exploit":
                a = self.exploit(s, w)

        a = check_act(a, self.action_dim)
        return a

    def calc_reward(self, s, w, episodicFeature=None):
        f = self.feature.extract(s)
        if episodicFeature is not None:
            episodicFeature += torch.linalg.norm(f, axis=0)
        r = torch.sum(w * f, 1)
        return r

    def explore(self):
        raise NotImplementedError

    def exploit(self):
        raise NotImplementedError

    def calc_priority_error(self):
        raise NotImplementedError

    def save_torch_model(self):
        raise NotImplementedError

    def load_torch_model(self):
        raise NotImplementedError


class MultitaskAgent(IsaacAgent):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.adaptive_task = self.env_cfg["task"]["adaptive_task"]

    def train_episode(self, gui_app=None, gui_rew=None):
        episode_r, _ = super().train_episode(gui_app=gui_app, gui_rew=gui_rew)

        if self.adaptive_task:
            self.task.adapt_task(episode_r)

    def act(self, s, task, mode="explore"):
        w = copy.copy(np2ts(task.W))
        id = copy.copy(np2ts(task.id))
        s = check_obs(s, self.observation_dim)

        with torch.no_grad():
            if (self.steps <= self.min_n_experience) and mode == "explore":
                a = 2 * torch.rand((self.n_env, self.env.num_act), device="cuda:0") - 1

            if mode == "explore":
                a = self.explore(s, w, id)
            elif mode == "exploit":
                a = self.exploit(s, w, id)

        a = check_act(a, self.action_dim)
        return a
