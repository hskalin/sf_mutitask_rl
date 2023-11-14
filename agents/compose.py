import isaacgym
from pathlib import Path

import torch
import wandb
from common.agent import IsaacAgent, RainbowAgent
from common.policy import MultiheadGaussianPolicy
from common.util import (
    grad_false,
    hard_update,
    pile_sa_pairs,
    soft_update,
    update_params,
)
from common.value_function import TwinnedMultiheadSFNetwork
from common.compositions import Compositions
from torch.optim import Adam
import numpy as np


class CompositionAgent(RainbowAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.lr = self.agent_cfg["lr"]
        self.policy_lr = self.agent_cfg["policy_lr"]
        self.value_net_kwargs = self.agent_cfg["value_net_kwargs"]
        self.policy_net_kwargs = self.agent_cfg["policy_net_kwargs"]
        self.gamma = self.agent_cfg["gamma"]
        self.tau = self.agent_cfg["tau"]
        
        self.td_target_update_interval = int(
            self.agent_cfg["td_target_update_interval"]
        )
        self.updates_per_step = self.agent_cfg["updates_per_step"]
        self.grad_clip = self.agent_cfg["grad_clip"]
        self.entropy_tuning = self.agent_cfg["entropy_tuning"]

        if self.agent_cfg.get("augmentHeads", True):
            self.pseudo_w = torch.tensor(
                self.env_cfg["task"]["taskSet_achievable"], device="cuda:0", dtype=torch.float32
            )
        else:
            self.pseudo_w = torch.eye(self.feature_dim).to(self.device)  # base tasks

        self.n_heads = self.pseudo_w.shape[0]
        self.droprate = self.value_net_kwargs["droprate"]

        self.sf = TwinnedMultiheadSFNetwork(
            observation_dim=self.observation_dim,
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            n_heads=self.n_heads,
            **self.value_net_kwargs,
        ).to(self.device)

        self.sf_target = TwinnedMultiheadSFNetwork(
            observation_dim=self.observation_dim,
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            n_heads=self.n_heads,
            **self.value_net_kwargs,
        ).to(self.device)
        if self.droprate <= 0.0:
            self.sf_target = self.sf_target.eval()

        hard_update(self.sf_target, self.sf)
        grad_false(self.sf_target)

        self.policy = MultiheadGaussianPolicy(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            n_heads=self.n_heads,
            **self.policy_net_kwargs,
        ).to(self.device)

        self.sf_optimizer = Adam(self.sf.parameters(), lr=self.lr, betas=[0.9, 0.999])
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=self.policy_lr, betas=[0.9, 0.999]
        )

        self.prev_impact = torch.zeros((self.n_env, self.n_heads, self.action_dim)).to(
            self.device
        )
        self.comp = Compositions(
            self.agent_cfg,
            self.policy,
            self.sf,
            self.prev_impact,
            self.n_heads,
            self.action_dim,
            self.feature_dim,
        )

        # masks used for vectorizing SFs head selction
        self.mask = torch.eye(self.n_heads, device="cuda:0").unsqueeze(dim=-1)
        self.mask = self.mask.repeat(self.mini_batch_size, 1, self.feature_dim)
        self.mask = self.mask.bool()

        if self.entropy_tuning:
            self.alpha_lr = self.agent_cfg["alpha_lr"]
            self.target_entropy = (
                -torch.prod(torch.tensor(self.action_shape, device=self.device))
                .tile(self.n_heads)
                .unsqueeze(1)
            )  # -|A|, [H,1]
            # # optimize log(alpha), instead of alpha
            self.log_alpha = torch.zeros(
                (self.n_heads, 1), requires_grad=True, device=self.device
            )
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = torch.tensor(self.agent_cfg["alpha"]).to(self.device)

        # init params
        self.learn_steps = 0

    def explore(self, s, w):
        # [N, A] <-- [N, S], [N, H, A], [N, F]
        a = self.comp.composition_fn(s, w, mode="explore")
        return a  # [N, A]

    def exploit(self, s, w):
        # [N, A] <-- [N, S], [N, H, A], [N, F]
        a = self.comp.composition_fn(s, w, mode="exploit")
        return a  # [N, A]

    def reset_env(self):
        self.prev_impact = torch.zeros((self.n_env, self.n_heads, self.action_dim)).to(
            self.device
        )
        self.comp.impact_x_idx = []
        self.comp.policy_idx = []
        return super().reset_env()

    def learn(self):
        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            soft_update(self.sf_target, self.sf, self.tau)

        batch = self.replay_buffer.sample(self.mini_batch_size)

        sf_loss, errors, mean_sf1, mean_sf2, target_sf = self.update_sf(batch)
        policy_loss, entropies, penalty_act = self.update_policy(batch)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies)
            update_params(self.alpha_optimizer, None, entropy_loss)
            self.alpha = self.log_alpha.exp()

        if self.learn_steps % self.log_interval == 0:
            metrics = {
                "loss/SF": sf_loss,
                "loss/policy": policy_loss,
                "loss/penalty_act": penalty_act,
                "state/mean_SF1": mean_sf1,
                "state/mean_SF2": mean_sf2,
                "state/target_sf": target_sf,
                "state/lr": self.lr,
                "state/entropy": entropies.detach().mean().item(),
                "state/policy_idx": wandb.Histogram(self.comp.policy_idx),
            }
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

    def update_sf(self, batch):
        (s, f, a, _, s_next, dones) = batch

        curr_sf1, curr_sf2 = self.sf(s, a)  # [N, H, F] <-- [N, S], [N,A]        
        target_sf = self.calc_target_sf(f, s_next, dones)  # [N, H, F]

        loss1 = (curr_sf1 - target_sf).pow(2).mean()  # R <-- [N, H, F]
        loss2 = (curr_sf2 - target_sf).pow(2).mean()  # R <-- [N, H, F]
        sf_loss = loss1 + loss2

        self.sf_optimizer.zero_grad(set_to_none=True)
        sf_loss.backward()
        self.sf_optimizer.step()

        # TD errors for updating priority weights
        errors = torch.mean(torch.abs(curr_sf1.detach() - target_sf), (1, 2))

        # update sf scale 
        self.comp.sf_norm_coeff_feat = curr_sf1.mean([0, 1]).abs()

        # log means to monitor training.
        sf_loss = sf_loss.detach().item()
        mean_sf1 = curr_sf1.detach().mean().item()
        mean_sf2 = curr_sf2.detach().mean().item()
        target_sf = target_sf.detach().mean().item()

        return sf_loss, errors, mean_sf1, mean_sf2, target_sf

    def update_policy(self, batch):
        (s, f, a, r, s_next, dones) = batch

        a_heads, entropies, _ = self.policy(s)  # [N,H,A], [N, H, 1] <-- [N,S]

        penalty_act = torch.norm(a_heads, p=1, dim=2, keepdim=True) / self.action_dim

        qs = self.calc_qs_from_sf(s, a_heads)

        qs = qs.unsqueeze(2)  # [N,H,1]

        loss = -qs - self.alpha * entropies  # + (1 - self.alpha) * penalty_act
        # [N, H, 1] <--  [N, H, 1], [N,1,1], [N, H, 1]
        policy_loss = torch.mean(loss)
        update_params(self.policy_optimizer, self.policy, policy_loss, self.grad_clip)

        return (
            policy_loss.detach().item(),
            entropies,
            penalty_act.mean().detach().item(),
        )

    def calc_entropy_loss(self, entropy):
        loss = self.log_alpha * (self.target_entropy - entropy).detach()
        # [N, H, 1] <--  [N, H, 1], [N,1,1]
        entropy_loss = -torch.mean(loss)
        return entropy_loss

    def calc_qs_from_sf(self, s, a):
        s_tiled, a_tiled = pile_sa_pairs(s, a)
        # [NHa, S], [NHa, A] <-- [N, S], [N, Ha, A]

        curr_sf1, curr_sf2 = self.sf(s_tiled, a_tiled)
        # [NHa, Hsf, F] <-- [NHa, S], [NHa, A]

        curr_sf = self.process_sfs(curr_sf1, curr_sf2)
        # [N, Ha, F] <-- [NHa, Hsf, F]

        qs = torch.einsum(
            "ijk,kj->ij", curr_sf, self.pseudo_w.T
        )  # [N,H]<-- [N,H,F]*[F, H]

        return qs

    def calc_target_sf(self, f, s_next, dones):
        _, _, a_next = self.policy(s_next)  # [N, H, 1],[N, H, A] <-- [N, S]

        with torch.no_grad():
            s_tiled, a_tiled = pile_sa_pairs(s_next, a_next)
            # [NHa, S], [NHa, A] <-- [N, S], [N, Ha, A]

            next_sf1, next_sf2 = self.sf_target(s_tiled, a_tiled)
            # [NHa, Hsf, F] <-- [NHa, S], [NHa, A]

            next_sf = self.process_sfs(next_sf1, next_sf2)
            # [N, Ha, F] <-- [NHa, Hsf, F]

        f = torch.tile(f[:, None, :], (self.n_heads, 1))  # [N,H,F] <-- [N,F]
        target_sf = f + torch.einsum(
            "ijk,il->ijk", next_sf, (~dones) * self.gamma
        )  # [N, H, F] <-- [N, H, F]+ [N, H, F]

        return target_sf  # [N, H, F]

    def calc_priority_error(self, batch):
        (s, f, a, _, s_next, dones) = batch

        with torch.no_grad():
            curr_sf1, curr_sf2 = self.sf(s, a)
            curr_sf = self.process_sfs(curr_sf1, curr_sf2)

        target_sf = self.calc_target_sf(f, s_next, dones)
        error = torch.mean(torch.abs(curr_sf - target_sf), (1, 2))
        return error.unsqueeze(1).cpu().numpy()

    def process_sfs(self, sf1, sf2):
        sf1, sf2 = self.mask_select_and_reshape_sfs(sf1, sf2)
        sf = self.calc_sf_from_double_sfs(sf1, sf2)
        return sf

    def mask_select_and_reshape_sfs(self, sf1, sf2):
        # [N, Ha, F] <-- [NHa, Hsf, F]
        sf1 = torch.masked_select(sf1, self.mask).view(
            self.mini_batch_size, self.n_heads, self.feature_dim
        )
        sf2 = torch.masked_select(sf2, self.mask).view(
            self.mini_batch_size, self.n_heads, self.feature_dim
        )
        return sf1, sf2

    def calc_sf_from_double_sfs(self, sf1, sf2):
        if self.droprate > 0.0:
            sf = 0.5 * (sf1 + sf2)
        else:
            sf = torch.min(sf1, sf2)  # [N, H, F]
        return sf

    def save_torch_model(self):
        path = self.log_path + f"model{self.episodes}/"
        print("saving model at ", path)
        Path(path).mkdir(parents=True, exist_ok=True)
        self.policy.save(path + "policy")
        self.sf.SF1.save(path + "sf1")
        self.sf.SF2.save(path + "sf2")

    def load_torch_model(self, path):
        self.policy.load(path + "policy")
        self.sf.SF1.load(path + "sf1")
        self.sf.SF2.load(path + "sf2")
        hard_update(self.sf_target, self.sf)
        grad_false(self.sf_target)
