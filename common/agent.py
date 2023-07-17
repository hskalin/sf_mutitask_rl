import datetime
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from env.wrapper.multiTask import MultiTaskEnv

import wandb
from common.helper import Visualizer

# from common.replay_buffer import (
#     MyMultiStepMemory,
#     MyPrioritizedMemory,
# )
from common.vec_buffer import VectorizedReplayBuffer
import os
from common.feature import pm_feature
from common.util import check_obs, check_act, dump_cfg, np2ts, to_batch, AverageMeter

warnings.simplefilter("once", UserWarning)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
exp_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

task_path = os.path.dirname(os.path.realpath(__file__)) + "/.."
log_path = task_path + "/../log/"


class IsaacAgent:
    def __init__(self, cfg):
        self.env_cfg = cfg["env"]
        self.agent_cfg = cfg["agent"]
        self.buffer_cfg = cfg["buffer"]

        self.device = cfg["rl_device"]

        env_spec = MultiTaskEnv(env_cfg=self.env_cfg)
        self.env, w, feature = env_spec.getEnv()
        self.env_max_steps = self.env_cfg["max_episode_length"]

        self.n_env = self.env_cfg["num_envs"]
        self.episode_max_step = self.env_cfg["episode_max_step"]
        self.render = self.env_cfg["render"]
        self.log_interval = self.env_cfg["log_interval"]
        self.total_episodes = int(self.env_cfg["total_episodes"])
        self.total_timesteps = self.n_env * self.episode_max_step * self.total_episodes

        self.eval = self.env_cfg["eval"]
        self.eval_interval = self.env_cfg["eval_interval"]
        self.eval_episodes = self.env_cfg["eval_episodes"]
        self.record = self.env_cfg["record"]
        self.save_model = self.env_cfg["save_model"]

        w_train, w_eval = w[0], w[1]  # [F]
        self.w_navi, self.w_hover = w_train[0], w_train[1]  # [F]
        self.w_eval_navi, self.w_eval_hover = w_eval[0], w_eval[1]  # [F]
        self.w_init = torch.tile(self.w_navi, (self.n_env, 1))  # [N, F]
        self.w_eval_init = torch.tile(self.w_eval_navi, (self.n_env, 1))  # [N, F]
        self.w = self.w_init.clone().type(torch.float32)  # [N, F]
        self.w_eval = self.w_eval_init.clone().type(torch.float32)  # [N, F]

        self.feature = feature
        self.observation_dim = self.env.num_obs
        self.feature_dim = self.feature.dim
        self.action_dim = self.env.num_act
        self.observation_shape = [self.observation_dim]
        self.feature_shape = [self.feature_dim]
        self.action_shape = [self.action_dim]

        self.per = self.buffer_cfg["prioritize_replay"]
        # memory = MyPrioritizedMemory if self.per else MyMultiStepMemory
        # self.replay_buffer = memory(
        #     state_shape=self.observation_shape,
        #     feature_shape=self.feature_shape,
        #     action_shape=self.action_shape,
        #     device=device,
        #     **self.buffer_cfg,
        # )
        self.replay_buffer = VectorizedReplayBuffer(
            self.observation_shape,
            self.action_shape,
            self.feature_shape,
            self.buffer_cfg["capacity"],
            self.device,
        )
        self.mini_batch_size = int(self.buffer_cfg["mini_batch_size"])
        self.min_n_experience = int(self.buffer_cfg["min_n_experience"])

        self.gamma = int(self.agent_cfg["gamma"])
        self.updates_per_step = int(self.agent_cfg["updates_per_step"])
        self.reward_scale = int(self.agent_cfg["reward_scale"])

        log_dir = (
            self.agent_cfg["name"]
            + "/"
            + self.env_cfg["env_name"]
            + "/"
            + exp_date
            + "/"
        )
        self.log_path = self.env_cfg["log_path"] + log_dir
        if self.record:
            Path(self.log_path).mkdir(parents=True, exist_ok=True)
            dump_cfg(self.log_path + "cfg", cfg)

        # self.visualizer = Visualizer(
        #     self.env,
        #     raisim_unity_path=self.env_cfg["raisim_unity_path"],
        #     render=self.render,
        #     record=self.record,
        #     save_video_path=self.log_path,
        # )

        self.steps = 0
        self.episodes = 0

        self.games_to_track = 100
        self.game_rewards = AverageMeter(1, self.games_to_track).to("cuda:0")
        self.game_lengths = AverageMeter(1, self.games_to_track).to("cuda:0")
        self.avgStepRew = AverageMeter(1, 20).to(self.device)

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.total_timesteps:
                break

    def train_episode(self, gui_app=None, gui_rew=None):
        self.episodes += 1
        episode_r = episode_steps = 0
        done = False

        print("episode = ", self.episodes)

        s = self.reset_env()
        for _ in range(self.episode_max_step):
            a = self.act(s)
            # _, done = self.env.step(a)

            self.env.step(a)
            done = self.env.reset_buf.clone()

            # episodeRet = self.env.return_buf.clone()
            episodeLen = self.env.progress_buf.clone()

            # s_next = self.env.observe(update_statistics=False)
            s_next = self.env.obs_buf.clone()
            self.env.reset()

            r = self.calc_reward(s_next, self.w)

            # call gui update loop
            if gui_app:
                gui_app.update_idletasks()
                gui_app.update()
                self.avgStepRew.update(r)
                gui_rew.set(self.avgStepRew.get_mean())

            masked_done = False if episode_steps >= self.episode_max_step else done
            self.save_to_buffer(s, a, r, s_next, done, masked_done)

            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            s = s_next
            self.w = self.w.float()
            self.update_w(s, self.w, self.w_navi, self.w_hover)

            self.steps += self.n_env
            episode_steps += 1
            episode_r += r

            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.size()[0]:
                self.game_rewards.update(episode_r[done_ids])
                self.game_lengths.update(episodeLen[done_ids])

            if episode_steps >= self.episode_max_step:
                break

        # if self.episodes % self.log_interval == 0:
        wandb.log({"reward/train": self.game_rewards.get_mean()})
        wandb.log({"length/train": self.game_lengths.get_mean()})

        if self.eval and (self.episodes % self.eval_interval == 0):
            self.evaluate()

    def update_w(self, s, w, w_navi, w_hover, thr=3):
        pos_index = self.env_cfg["feature"]["pos_index"]
        dim = self.env_cfg["dim"]
        dist = torch.linalg.norm(s[:, pos_index : pos_index + dim], axis=1)
        w[torch.where(dist <= thr)[0], :] = w_hover.float()
        w[torch.where(dist > thr)[0], :] = w_navi.float()

    def is_update(self):
        return (
            len(self.replay_buffer) > self.mini_batch_size
            and self.steps >= self.min_n_experience
        )

    def reset_env(self):
        # s = self.env.reset()
        s = self.env.obs_buf.clone()
        if s is None:
            s = torch.zeros((self.n_env, self.env.num_obs))

        self.w = self.w_init.clone()
        self.w_eval = self.w_eval_init.clone()
        return s

    def save_to_buffer(self, s, a, r, s_next, done, masked_done):
        f = self.feature.extract(s)

        r = r[:, None] * self.reward_scale
        done = done[:, None]
        masked_done = masked_done[:, None]

        # if self.per:
        #     error = self.calc_priority_error(
        #         to_batch(s, f, a, r, s_next, masked_done, device)
        #     )
        #     self.replay_buffer.append(s, f, a, r, s_next, masked_done, error, done)
        # else:
        self.replay_buffer.add(s, f, a, r, s_next, masked_done)

    def evaluate(self):
        episodes = int(self.eval_episodes)
        if episodes == 0:
            return

        print(f"===== evaluate at episode: {self.episodes} ====")
        print(f"===== eval for running for {self.env_max_steps} steps ===")
        # self.visualizer.turn_on(self.episodes)

        returns = torch.zeros((episodes,), dtype=torch.float32)
        for i in range(episodes):
            episode_r = 0.0

            s = self.reset_env()
            for _ in range(self.env_max_steps):
                a = self.act(s, "exploit")
                self.env.step(a)
                # s_next = self.env.observe(update_statistics=False)
                s_next = self.env.obs_buf.clone()
                self.env.reset()

                r = self.calc_reward(s_next, self.w_eval)

                s = s_next
                self.w_eval = self.w_eval.float()
                self.update_w(s, self.w_eval, self.w_eval_navi, self.w_eval_hover)
                episode_r += r

                if self.render:
                    time.sleep(0.04)

            returns[i] = torch.mean(episode_r).item()

        print(f"===== finish evaluate ====")
        # self.visualizer.turn_off()
        wandb.log({"reward/eval": torch.mean(returns).item()})

        if self.save_model:
            self.save_torch_model()

    def act(self, s, mode="explore"):
        if self.steps <= self.min_n_experience:
            a = 2 * torch.rand((self.n_env, self.env.num_act), device="cuda:0") - 1
        else:
            a = self.get_action(s, mode)

        a = check_act(a, self.action_dim)
        return a

    def get_action(self, s, mode):
        s, w = np2ts(s), np2ts(self.w)
        s = check_obs(s, self.observation_dim)

        with torch.no_grad():
            if mode == "explore":
                a = self.explore(s, w)
            elif mode == "exploit":
                a = self.exploit(s, w)
        return a

    def calc_reward(self, s, w):
        f = self.feature.extract(s)
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
