from pathlib import Path

import isaacgym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.agent import MultitaskAgent
from common.compositions import Compositions
from common.feature_extractor import TCN
from common.policy import MultiheadGaussianPolicy
from common.util import (
    grad_false,
    hard_update,
    pile_sa_pairs,
    soft_update,
    update_params,
)
from common.value import MultiheadSFNetwork
from torch.optim import Adam

import wandb


class ENVEncoderBuilder(nn.Module):
    def __init__(self, in_dim, hidden_dim) -> None:
        super().__init__()

        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, in_dim)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

    def forward(self, xu):
        x = self.ln1(xu)
        x = F.relu(self.l1(xu))
        x = torch.cat([x, xu], dim=1)

        x = self.ln2(x)
        x = F.relu(self.l2(x))
        x = torch.cat([x, xu], dim=1)

        x = self.ln3(x)
        x = self.l3(x)
        return x


class ENVEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim) -> None:
        self.model = torch.jit.trace(ENVEncoderBuilder(in_dim, out_dim, hidden_dim))

    def forward(self, x):
        return self.model(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class AdaptationModule(nn.Module):
    def __init__(self, in_dim, out_dim, stack_size, kernel_size=5) -> None:
        super().__init__()
        self.tcn = TCN(
            in_dim=in_dim,
            out_dim=out_dim,
            num_channels=[in_dim, in_dim, in_dim],
            stack_size=stack_size,
            kernel_size=kernel_size,
        )

    def forward(self, x):
        return self.tcn(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class FIFOBuffer:
    def __init__(self, n_env, traj_dim, stack_size, device) -> None:
        self.n_env = n_env
        self.traj_dim = traj_dim
        self.stack_size = stack_size
        self.device = device
        self.buf = torch.empty(stack_size, n_env, traj_dim).to(device)

    def add(self, data):
        self.buf = torch.cat([self.buf[1:], data[None, ...]])

    def get(self):
        buf = self.buf.permute(1, 2, 0)
        buf = torch.flip(buf, (2,))
        return buf

    def clear(self):
        self.buf = torch.empty(self.stack_size, self.n_env, self.traj_dim).to(
            self.device
        )


class RMACompAgent(MultitaskAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.framestacked_replay = self.buffer_cfg["framestacked_replay"]
        self.stack_size = self.buffer_cfg["stack_size"]
        assert (
            self.framestacked_replay == True
        ), "This agent only support framestacked replay"

        self.lr = self.agent_cfg["lr"]
        self.policy_lr = self.agent_cfg["policy_lr"]
        self.lr_schedule = self.agent_cfg["lr_schedule"]
        self.sf_net_kwargs = self.agent_cfg["sf_net_kwargs"]
        self.policy_net_kwargs = self.agent_cfg["policy_net_kwargs"]
        self.gamma = self.agent_cfg["gamma"]
        self.tau = self.agent_cfg["tau"]

        self.td_target_update_interval = int(
            self.agent_cfg["td_target_update_interval"]
        )
        self.updates_per_step = self.agent_cfg["updates_per_step"]
        self.grad_clip = self.agent_cfg["grad_clip"]
        self.entropy_tuning = self.agent_cfg["entropy_tuning"]
        self.prioritized_replay = self.buffer_cfg["prioritized_replay"]

        self.explore_method = self.agent_cfg.get("explore_method", "null")
        self.exploit_method = self.agent_cfg.get("exploit_method", "sfgpi")

        self.env_latent_dim = self.env.latent_dim
        self.env_latent_idx = self.env.latent_idx

        # rma
        # [phase1: train. phase2: train adaptation module. phase3: deploy]
        self.phase = 1
        self.rma = self.agent_cfg.get("rma", False)
        if self.rma:
            self.episodes_phase2 = self.env_cfg["episodes_phase2"]
            self.phase2_timesteps = (
                self.n_env * self.episode_max_step * self.episodes_phase2
            )

        # define primitive tasks
        self.augment_heads = self.agent_cfg.get("augment_heads", False)
        if self.augment_heads:
            self.w_primitive = self.task.Train.taskSet
        else:
            self.w_primitive = torch.eye(self.feature_dim).to(self.device)

        self.n_heads = self.w_primitive.shape[0]

        # define models
        self.latent_dim = self.observation_dim
        self.sf = MultiheadSFNetwork(
            observation_dim=self.observation_dim + self.latent_dim + self.action_dim,
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            n_heads=self.n_heads,
            **self.sf_net_kwargs,
        ).to(self.device)

        self.sf_target = MultiheadSFNetwork(
            observation_dim=self.observation_dim + self.latent_dim + self.action_dim,
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            n_heads=self.n_heads,
            **self.sf_net_kwargs,
        ).to(self.device)

        hard_update(self.sf_target, self.sf)
        grad_false(self.sf_target)

        self.policy = MultiheadGaussianPolicy(
            observation_dim=self.observation_dim + self.latent_dim + self.action_dim,
            action_dim=self.action_dim,
            n_heads=self.n_heads,
            **self.policy_net_kwargs,
        ).to(self.device)

        self.encoder = ENVEncoder(
            in_dim=self.env_latent_dim, out_dim=self.latent_dim, hidden_dim=256
        ).to(self.device)
        self.adaptor = AdaptationModule(
            in_dim=self.observation_dim + self.action_dim,
            out_dim=self.latent_dim,
            stack_size=self.stack_size - 1,
        ).to(self.device)

        self.sf_optimizer = Adam(
            [
                {"params": self.sf.parameters()},
                {"params": self.encoder.parameters()},
            ],
            lr=self.lr,
            betas=[0.9, 0.999],
        )
        self.policy_optimizer = Adam(
            [
                {"params": self.policy.parameters()},
                {"params": self.encoder.parameters()},
            ],
            lr=self.policy_lr,
            betas=[0.9, 0.999],
        )
        self.adaptor_optimizer = Adam(
            self.adaptor.parameters(), lr=self.lr, betas=[0.9, 0.999]
        )

        if self.lr_schedule:
            self.lrScheduler_sf = torch.optim.lr_scheduler.LinearLR(
                self.sf_optimizer,
                start_factor=1,
                end_factor=0.1,
                total_iters=self.episode_max_step * self.total_episodes,
            )
            self.lrScheduler_policy = torch.optim.lr_scheduler.LinearLR(
                self.policy_optimizer,
                start_factor=1,
                end_factor=0.1,
                total_iters=self.episode_max_step * self.total_episodes,
            )

        self.comp = Compositions(
            self.agent_cfg,
            self.policy,
            self.sf,
            self.n_env,
            self.n_heads,
            self.action_dim,
            self.feature_dim,
            self.device,
        )

        self._create_sf_mask()

        if self.entropy_tuning:
            self._create_entropy_tuner()
        else:
            self.alpha = torch.tensor(self.agent_cfg["alpha"]).to(self.device)

        # init params
        self.learn_steps = 0
        self.prev_a = torch.empty(self.n_env, self.action_dim)
        self.prev_traj = FIFOBuffer(
            n_env=self.n_env,
            traj_dim=self.observation_dim + self.action_dim,
            stack_size=self.stack_size,
        )

    def run(self):
        while self.phase == 1:
            print(f"============= start phase {self.phase} =============")
            self.train_episode()

            if self.eval and (self.episodes % self.eval_interval == 0):
                self.evaluate()
                if self.save_model:
                    self.save_torch_model()

            if self.steps > self.total_timesteps:
                break

        if self.rma:
            self.phase = 2
            grad_false(self.sf)
            grad_false(self.sf_target)
            grad_false(self.policy)
            grad_false(self.encoder)

            while self.phase == 2:
                print(f"============= start phase {self.phase} =============")
                self.train_episode()

                if self.eval and (self.episodes % self.eval_interval == 0):
                    self.evaluate()
                    if self.save_model:
                        self.save_torch_model()

                self.steps = 0
                if self.steps > self.phase2_timesteps:
                    break

        self.phase = 3
        print(f"============= finish =============")

    def explore(self, s, w, id):
        s_raw, e = self.parse_state(s)

        if self.phase == 1:
            z = self.encoder(e)
        else:
            z = self.adaptor(self.prev_traj.get())

        s = torch.concat([s_raw, z, self.prev_a], dim=1)

        # [N, A] <-- [N, S+Z+A], [N, F]
        a = self.comp.act(s, w, id, "explore", self.explore_method)

        self.prev_a = a
        if self.phase is not 1:
            self.prev_traj.add(torch.concat([s_raw, a], dim=1))
        return a

    def exploit(self, s, w, id):
        s_raw, e = self.parse_state(s)

        if self.phase == 1:
            z = self.encoder(e)
        else:
            z = self.adaptor(self.prev_traj.get())

        s = torch.concat([s_raw, z, self.prev_a], dim=1)

        # [N, A] <-- [N, S+Z+A], [N, F]
        a = self.comp.act(s, w, id, "exploit", self.exploit_method)

        self.prev_a = a
        if self.phase is not 1:
            self.prev_traj.add(torch.concat([s_raw, a], dim=1))
        return a  # [N, A]

    def parse_state(self, s):
        s, e = s[:, : self.env_latent_idx], s[:, self.env_latent_idx :]
        return s, e  # [N, S+Z+A], [N,S], [N, Z]

    def reset_env(self):
        self.comp.reset()
        self.prev_a = torch.empty(self.n_env, self.action_dim)
        self.prev_traj.clear()
        return super().reset_env()

    def add_task(self, w):
        w = w.view(-1, self.feature_dim)
        ntask = w.shape[0]

        # update submodules
        self.task.add_task(w)
        self.sf.add_head(ntask)
        self.sf_target.add_head(ntask)
        self.policy.add_head(ntask)
        self.comp.add_head(ntask)

        # update module
        self.n_heads += ntask
        self.w_primitive = self.task.Train.taskSet
        self._create_sf_mask()

        if self.entropy_tuning:
            self._create_entropy_tuner()

    def learn(self):
        if self.phase == 1:
            self.learn_phase1()
        else:
            self.learn_phase2()

    def learn_phase1(self):
        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            soft_update(self.sf_target, self.sf, self.tau)

        batch = self.replay_buffer.sample(self.mini_batch_size)

        if self.prioritized_replay:
            weights = batch["_weight"]
            weights = weights[:, None, None]
        else:
            weights = 1

        sf_loss, errors, mean_sf1, mean_sf2, target_sf = self.update_sf(batch, weights)
        policy_loss, entropies, action_norm = self.update_policy(batch, weights)

        if self.lr_schedule:
            self.lrScheduler_sf.step()
            self.lrScheduler_policy.step()

        if self.entropy_tuning:
            entropy_loss = self._calc_entropy_loss(entropies, weights)
            update_params(self.alpha_optimizer, None, entropy_loss)
            self.alpha = self.log_alpha.exp()

        if self.prioritized_replay:
            batch.set("td_error", errors)
            self.replay_buffer.update_tensordict_priority(batch)

        # start logging
        if self.learn_steps % self.log_interval == 0:
            metrics = {
                "loss/SF": sf_loss,
                "loss/policy": policy_loss,
                "loss/action_norm": action_norm,
                "state/mean_SF1": mean_sf1,
                "state/mean_SF2": mean_sf2,
                "state/target_sf": target_sf,
                "state/entropy": entropies.detach().mean().item(),
                "state/policy_idx": wandb.Histogram(self.comp.policy_idx),
            }

            if self.lr_schedule:
                metrics.update(
                    {
                        "state/lr_sf": self.sf_optimizer.param_groups[0]["lr"],
                        "state/lr_policy": self.policy_optimizer.param_groups[0]["lr"],
                    }
                )
            else:
                metrics.update(
                    {
                        "state/lr_sf": self.lr,
                        "state/lr_policy": self.policy_lr,
                    }
                )

            if self.comp.record_impact:
                metrics.update(
                    {
                        "state/impact_x_idx": wandb.Histogram(self.comp.impact_x_idx),
                    }
                )

            if self.entropy_tuning:
                metrics.update(
                    {
                        "loss/alpha": entropy_loss.detach().item(),
                        "state/alpha": self.alpha.mean().detach().item(),
                    }
                )

            wandb.log(metrics)

    def learn_phase2(self):
        self.learn_steps += 1
        batch = self.replay_buffer.sample(self.mini_batch_size)
        adaptor_loss = self.update_adaptor(batch)
        if self.learn_steps % self.log_interval == 0:
            metrics = {"loss/adaptor": adaptor_loss}
            wandb.log(metrics)

    def update_sf(self, batch, weights):
        (s, _, f, a, a_stack, _, s_next, dones) = (
            batch["obs"],
            batch["stacked_obs"],
            batch["feature"],
            batch["action"],
            batch["stacked_act"],
            batch["reward"],
            batch["next_obs"],
            batch["done"],
        )
        prev_a = a_stack[:, :, 1]  # [N, A] <-- [N, A, K]

        s, e = self.parse_state(s)
        z = self.encoder(e)
        s = torch.concat([s, z, prev_a], dim=1)

        s_next, e_next = self.parse_state(s_next)
        z_next = self.encoder(e_next)
        s_next = torch.concat([s_next, z_next, a], dim=1)  # [N, S+Z+A]

        curr_sf1, curr_sf2 = self.sf(s, a)  # [N, H, F] <--[N, S+Z+A], [N,A]
        target_sf = self._calc_target_sf(f, s_next, dones)  # [N, H, F]

        loss1 = torch.mean((curr_sf1 - target_sf).pow(2) * weights)  # R <-- [N, H, F]
        loss2 = torch.mean((curr_sf2 - target_sf).pow(2) * weights)  # R <-- [N, H, F]

        sf_loss = loss1 + loss2

        self.sf_optimizer.zero_grad(set_to_none=True)
        sf_loss.backward()
        self.sf_optimizer.step()

        # TD errors for updating priority weights
        errors = torch.mean(torch.abs(curr_sf1.detach() - target_sf), (1, 2))

        # update sf scale
        self.comp.update_sf_norm(curr_sf1.mean([0, 1]).abs())

        # log means to monitor training.
        sf_loss = sf_loss.detach().item()
        curr_sf1 = curr_sf1.detach().mean().item()
        curr_sf2 = curr_sf2.detach().mean().item()
        target_sf = target_sf.detach().mean().item()

        return sf_loss, errors, curr_sf1, curr_sf2, target_sf

    def update_policy(self, batch, weights):
        s, a_stack = batch["obs"], batch["stacked_act"]

        s_raw, e = self.parse_state(s)  # [N, S], [N, E]
        prev_a = a_stack[:, :, 1]  # [N, A] <-- [N, A, K]

        z = self.encoder(e)
        s = torch.concat([s_raw, z, prev_a], dim=1)

        # [N,H,A], [N, H, 1] <-- [N,S+Z+A]
        a_heads, entropies, _ = self.policy.sample(s)

        action_norm = torch.norm(a_heads, p=1, dim=2, keepdim=True) / self.action_dim

        qs = self._calc_qs_from_sf(s, a_heads)  # [N,H]<--[N, S+Z+A], [N, H, A]
        qs = qs.unsqueeze(2)  # [N,H,1]

        loss = -qs - self.alpha * entropies  # + (1 - self.alpha) * action_norm

        # [N, H, 1] <--  [N, H, 1], [N,1,1], [N, H, 1]
        policy_loss = torch.mean(loss * weights)
        update_params(self.policy_optimizer, self.policy, policy_loss, self.grad_clip)

        return (
            policy_loss.detach().item(),
            entropies,
            action_norm.mean().detach().item(),
        )

    def update_adaptor(self, batch):
        s_stack, a_stack = batch["stacked_obs"], batch["stacked_act"]

        s = s_stack[:, :, 0]  # [N, S+E] <-- [N, S+E, K]
        _, e = self.parse_state(s)  # [N, S], [N, E]

        prev_acts = a_stack[:, :, 1:]  # [N, A, K-1]
        prev_obses = s_stack[:, :, 1:]  # [N, S+E, K-1]
        prev_obses, _ = self.parse_state(prev_obses)  # [N, S, K-1]
        prev_traj = torch.concat([prev_acts, prev_obses], dim=1)  # [N, S+A, K-1]

        z = self.encoder(e)
        z_head = self.adaptor(prev_traj)

        adaptor_loss = torch.nn.MSELoss(z_head, z)

        update_params(
            self.adaptor_optimizer, self.adaptor, adaptor_loss, self.grad_clip
        )
        return adaptor_loss.detach().item()

    def _calc_entropy_loss(self, entropy, weights):
        loss = self.log_alpha * (self.target_entropy - entropy).detach()

        # [N, H, 1] <--  [N, H, 1], [N,1,1]
        entropy_loss = -torch.mean(loss * weights)
        return entropy_loss

    def _calc_qs_from_sf(self, s, a):
        # [NHa, S], [NHa, A] <-- [N, S], [N, Ha, A]
        s_tiled, a_tiled = pile_sa_pairs(s, a)

        # [NHa, Hsf, F] <-- [NHa, S], [NHa, A]
        curr_sf1, curr_sf2 = self.sf(s_tiled, a_tiled)

        # [N, Ha, F] <-- [NHa, Hsf, F]
        curr_sf = self._process_sfs(curr_sf1, curr_sf2)

        # [N,H]<-- [N,H,F]*[H,F]
        qs = torch.einsum("ijk,jk->ij", curr_sf, self.w_primitive)

        return qs

    def _calc_target_sf(self, f, s, dones):
        _, _, a = self.policy.sample(s)  # [N, H, A] <-- [N, S+Z+A]

        with torch.no_grad():
            # [NHa, S+Z+A], [NHa, A] <-- [N, S+Z+A], [N, Ha, A]
            s_tiled, a_tiled = pile_sa_pairs(s, a)

            # [NHa, Hsf, F] <-- [NHa, S+Z+A], [NHa, A]
            next_sf1, next_sf2 = self.sf_target(s_tiled, a_tiled)

            # [N, Ha, F] <-- [NHa, Hsf, F]
            next_sf = self._process_sfs(next_sf1, next_sf2)

        # [N,H,F] <-- [N,F]
        f = torch.tile(f[:, None, :], (self.n_heads, 1))

        # [N, H, F] <-- [N, H, F] + [N, H, F]
        target_sf = f + torch.einsum("ijk,il->ijk", next_sf, (~dones) * self.gamma)

        return target_sf  # [N, H, F]

    def _process_sfs(self, sf1, sf2):
        sf = torch.min(sf1, sf2)  # [NHa, Hsf, F]

        # [NHa, F] <-- [NHa, Hsf, F], [NHa, Hsf, F]
        sf = torch.masked_select(sf, self.mask)

        # [N, Ha, F] <-- [NHa, F]
        sf = sf.view(self.mini_batch_size, self.n_heads, self.feature_dim)
        return sf

    def _create_entropy_tuner(self):
        self.alpha_lr = self.agent_cfg["alpha_lr"]
        self.target_entropy = (
            -torch.prod(torch.tensor(self.action_shape, device=self.device))
            .tile(self.n_heads)
            .unsqueeze(1)
        )  # -|A|, [H,1]

        # optimize log(alpha), instead of alpha
        if hasattr(self, "log_alpha"):
            self.log_alpha = torch.concat(
                [self.log_alpha.data, torch.zeros([1, 1])]
            ).to(self.device)
            self.log_alpha.requires_grad = True
        else:
            self.log_alpha = torch.zeros(
                (self.n_heads, 1), requires_grad=True, device=self.device
            )

        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.alpha_lr)

    def _create_sf_mask(self):
        # masks used for vectorizing SFs head selction
        # [H, H, 1]
        self.mask = torch.eye(self.n_heads, device=self.device).unsqueeze(dim=-1)
        # [NH, H, F]
        self.mask = self.mask.repeat(self.mini_batch_size, 1, self.feature_dim)
        self.mask = self.mask.bool()

    def save_torch_model(self):
        path = self.log_path + f"model{self.episodes}/"

        print("saving model at ", path)
        Path(path).mkdir(parents=True, exist_ok=True)
        self.policy.save(path + "policy")
        self.sf.save(path + "sf")
        self.encoder.save(path + "encoder")
        self.adaptor.save(path + "adaptor")

    def load_torch_model(self, path):
        self.policy.load(path + "policy")
        self.sf.load(path + "sf")
        self.encoder.load(path + "encoder")
        self.adaptor.load(path + "adaptor")

        hard_update(self.sf_target, self.sf)
        grad_false(self.sf_target)
