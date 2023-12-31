# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy
import argparse
import os
import random
import time
from distutils.util import strtobool
from env.wrapper.multiTask import multitaskenv_constructor
from common.util import AverageMeter
from common.feature_extractor import TCN, Perception

import gym
import wandb

from env import env_map

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(env.num_obs, 400)),
            nn.Tanh(),
            layer_init(nn.Linear(400, 400)),
            nn.Tanh(),
            layer_init(nn.Linear(400, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(env.num_obs, 400)),
            nn.Tanh(),
            layer_init(nn.Linear(400, 400)),
            nn.Tanh(),
            layer_init(nn.Linear(400, env.num_act), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, env.num_act))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )

class TCNAgent(Agent):
    def __init__(self, env):
        super().__init__(env)

        channels = [env.num_obs, env.num_obs]
        kernel_size = 2
        self.tcn1 = TCN(
            in_dim = env.num_obs,
            out_dim = env.num_obs,
            num_channels=channels,
            kernel_size=kernel_size,
        )
        self.tcn2 = TCN(
            in_dim = env.num_obs,
            out_dim = env.num_obs,
            num_channels=channels,
            kernel_size=kernel_size,
        )

    def get_value(self, x):
        x1 = self.tcn1(x)
        return self.critic(x1)

    def get_action_and_value(self, x, action=None):
        x1 = self.tcn1(x)
        x2 = self.tcn2(x)
        action_mean = self.actor_mean(x2)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x1),
        )


class PPO_agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env_cfg = cfg["env"]
        self.agent_cfg = cfg["agent"]
        self.buffer_cfg = cfg["buffer"]

        self.num_envs = self.env_cfg["num_envs"]
        self.num_steps = self.agent_cfg["num_steps"]

        self.agent_cfg["batch_size"] = int(
            self.env_cfg["num_envs"] * self.agent_cfg["num_steps"]
        )
        self.agent_cfg["minibatch_size"] = int(
            self.agent_cfg["batch_size"] // self.agent_cfg["num_minibatches"]
        )

        self.device = cfg["rl_device"]

        self.env, _, _ = multitaskenv_constructor(env_cfg=self.env_cfg, device=self.device)

        self.run_name = (
            f"{self.env_cfg['env_name']}__{self.agent_cfg['name']}__{int(time.time())}"
        )

        self.writer = SummaryWriter(f"runs/{self.run_name}")

        #self.agent = Agent(self.env).to(self.device)
        self.history = 5
        self.agent = TCNAgent(self.env).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.agent_cfg["learning_rate"], eps=1e-5
        )
        
        self.game_rewards = AverageMeter(1, max_size=100).to(self.device)
        self.game_lengths = AverageMeter(1, max_size=100).to(self.device)

        self._init_buffers()
        ###

    def _init_buffers(self):
        # ALGO Logic: Storage setup
        self.obs = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"], self.env.num_obs),
            dtype=torch.float,
        ).to(self.device)
        self.actions = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"], self.env.num_act),
            dtype=torch.float,
        ).to(self.device)
        self.logprobs = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"]), dtype=torch.float
        ).to(self.device)
        self.rewards = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"]), dtype=torch.float
        ).to(self.device)
        self.dones = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"]), dtype=torch.float
        ).to(self.device)
        self.values = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"]), dtype=torch.float
        ).to(self.device)
        self.advantages = torch.zeros_like(self.rewards, dtype=torch.float).to(
            self.device
        )

    def update(self):
        raise NotImplemented

    def run(self):
        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()

        # next_obs = envs.reset()
        next_obs = self.env.obs_buf

        next_done = torch.zeros(self.env_cfg["num_envs"], dtype=torch.float).to(
            self.device
        )
        num_updates = self.agent_cfg["total_timesteps"] // self.agent_cfg["batch_size"]

        for update in range(1, num_updates + 1):
            wandb_metrics = {}
            # Annealing the rate if instructed to do so.
            if self.agent_cfg["anneal_lr"]:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.agent_cfg["learning_rate"]
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.agent_cfg["num_steps"]):
                global_step += 1 * self.env_cfg["num_envs"]
                self.obs[step] = next_obs
                self.dones[step] = next_done

                prev_idx = max(0, step - self.history)
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(
                        #next_obs.unsqueeze(-1)
                        self.obs.permute(1,2,0)[...,prev_idx:step+1]
                    )
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.

                # next_obs, rewards[step], next_done, info = envs.step(action)

                self.env.step(action)
                next_obs, self.rewards[step], next_done, episodeLen, episodeRet = (
                    self.env.obs_buf,
                    self.env.reward_buf,
                    self.env.reset_buf.clone(),
                    self.env.progress_buf.clone(),
                    self.env.return_buf.clone(),
                )
                self.env.reset()

                # if 0 <= step <= 2:
                done_ids = next_done.nonzero(as_tuple=False).squeeze(-1)
                if done_ids.size()[0]:
                    # taking mean over all envs that are done at the
                    # current timestep
                    # episodic_return = torch.mean(episodeRet[done_ids].float()).item()
                    # episodic_length = torch.mean(episodeLen[done_ids].float()).item()

                    self.game_rewards.update(episodeRet[done_ids])
                    self.game_lengths.update(episodeLen[done_ids])

                    episodic_return = self.game_rewards.get_mean() 
                    episodic_length = self.game_lengths.get_mean()
                    
                    print(
                        f"global_step={global_step}, episodic_return={episodic_return}"
                    )
                    wandb_metrics.update(
                        {
                            "rewards/step": episodic_return,
                            "episode_lengths/step": episodic_length,
                        }
                    )
                    self.writer.add_scalar("rewards/step", episodic_return, global_step)
                    self.writer.add_scalar(
                        "episode_lengths/step", episodic_length, global_step
                    )

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(
                    #next_obs.unsqueeze(-1)
                    self.obs.permute(1,2,0)[...,-self.history:]
                    ).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.agent_cfg["num_steps"])):
                    if t == self.agent_cfg["num_steps"] - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = (
                        self.rewards[t]
                        + self.agent_cfg["gamma"] * nextvalues * nextnonterminal
                        - self.values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.agent_cfg["gamma"]
                        * self.agent_cfg["gae_lambda"]
                        * nextnonterminal
                        * lastgaelam
                    )
                returns = advantages + self.values

            # flatten the batch
            #b_obs = self.obs.reshape((-1, self.env.num_obs))
            unfold_obs = self.obs.permute(1,2,0).unfold(2,self.history,1)

            n_batch = self.num_envs*(self.num_steps - self.history+1)
            b_obs = unfold_obs.permute(0,2,1,3).reshape(
                n_batch, self.env.num_obs, self.history) # [Nenv*(S-H+1), O, H]
            b_logprobs = self.logprobs.reshape(-1)[-n_batch:]
            b_actions = self.actions.reshape((-1, self.env.num_act))[-n_batch:]
            b_advantages = advantages.reshape(-1)[-n_batch:]
            b_returns = returns.reshape(-1)[-n_batch:]
            b_values = self.values.reshape(-1)[-n_batch:]

            # Optimizing the policy and value network
            clipfracs = []
            for epoch in range(self.agent_cfg["update_epochs"]):
                # b_inds = torch.randperm(
                #     self.agent_cfg["batch_size"], device=self.device
                # )
                b_inds = torch.randperm(
                    self.num_envs*(self.num_steps - self.history+1), device=self.device
                )
                for start in range(
                    0, self.agent_cfg["batch_size"], self.agent_cfg["minibatch_size"]
                ):
                    end = start + self.agent_cfg["minibatch_size"]
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.agent_cfg["clip_coef"])
                            .float()
                            .mean()
                            .item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.agent_cfg["norm_adv"]:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio,
                        1 - self.agent_cfg["clip_coef"],
                        1 + self.agent_cfg["clip_coef"],
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.agent_cfg["clip_vloss"]:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.agent_cfg["clip_coef"],
                            self.agent_cfg["clip_coef"],
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - self.agent_cfg["ent_coef"] * entropy_loss
                        + v_loss * self.agent_cfg["vf_coef"]
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.agent_cfg["max_grad_norm"]
                    )
                    self.optimizer.step()

                if self.agent_cfg["target_kl"] is not None:
                    if approx_kl > self.agent_cfg["target_kl"]:
                        break

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar(
                "charts/learning_rate",
                self.optimizer.param_groups[0]["lr"],
                global_step,
            )
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar(
                "losses/old_approx_kl", old_approx_kl.item(), global_step
            )
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            self.writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )
            wandb_metrics.update(
                {
                    "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                }
            )

            wandb.log(wandb_metrics)


        # envs.close()
        self.writer.close()

    def test(self):
        raise NotImplemented
