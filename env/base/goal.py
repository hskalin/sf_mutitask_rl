import torch
import math
import numpy as np


class RandomWayPoints:
    """a random generated goal during training"""

    def __init__(
        self,
        device,
        num_envs,
        init_pos=None,
        init_vel=None,
        init_ang=None,
        init_angvel=None,
        rand_pos=True,
        rand_ang=True,
        rand_vel=True,
        rand_angvel=True,
        pos_lim=None,
        vel_lim=None,
        angvel_lim=None,
        kWayPt=1,
        wp_max_dist=10 / np.sqrt(3),  # [m] generate next wp within range
        trigger_dist=2,  # [m] activate next wp if robot within range
        min_z=5,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.kWayPt = kWayPt

        self.pos_lim = pos_lim
        self.vel_lim = vel_lim
        self.angvel_lim = angvel_lim

        self.rand_pos = rand_pos
        self.rand_ang = rand_ang
        self.rand_vel = rand_vel
        self.rand_angvel = rand_angvel

        self.wp_max_dist = wp_max_dist
        self.trigger_dist = torch.tensor(trigger_dist).to(self.device)
        self.min_z = min_z

        if init_pos is not None:
            self.init_pos = init_pos
            self.pos = torch.tile(
                torch.tensor(init_pos, device=self.device, dtype=torch.float32),
                (self.num_envs, self.kWayPt, 1),
            )
        else:
            self.pos = None

        if init_vel is not None:
            self.init_vel = init_vel
            self.vel = torch.tile(
                torch.tensor(init_vel, device=self.device, dtype=torch.float32),
                (self.num_envs, self.kWayPt, 1),
            )
        else:
            self.vel = None

        if init_ang is not None:
            self.init_ang = init_ang
            self.ang = torch.tile(
                torch.tensor(init_ang, device=self.device, dtype=torch.float32),
                (self.num_envs, 1),
            )
        else:
            self.ang = None

        if init_angvel is not None:
            self.init_angvel = init_angvel
            self.angvel = torch.tile(
                torch.tensor(init_angvel, device=self.device, dtype=torch.float32),
                (self.num_envs, 1),
            )
        else:
            self.angvel = None

        self.idx = torch.zeros(self.num_envs, 1, device=self.device).to(torch.long)

    def get_pos(self):
        return self.pos[range(self.num_envs), self.idx.squeeze()]

    def get_vel(self):
        return self.vel[range(self.num_envs), self.idx.squeeze()]

    def update_idx(self, rb_pos):
        """check if robot is close to waypoint"""
        dist = torch.norm(
            rb_pos - self.pos[range(self.num_envs), self.idx.squeeze()],
            p=2,
            dim=1,
            keepdim=True,
        )
        self.idx = torch.where(dist <= self.trigger_dist, self.idx + 1, self.idx)
        self.idx = torch.where(self.idx > self.kWayPt - 1, 0, self.idx)
        print(self.idx.squeeze())

    def sample(self, env_ids):
        if self.rand_pos:
            self.pos[env_ids] = self._sample_within_distance(
                len(env_ids), self.pos_lim, kWP=self.kWayPt, dist=self.wp_max_dist
            )

        if self.rand_vel:
            self.vel[env_ids] = self._sample(
                (len(env_ids), self.kWayPt, 3), self.vel_lim
            )

        if self.rand_ang:
            self.ang[env_ids] = self._sample((len(env_ids), 3), math.pi)

        if self.rand_angvel:
            self.angvel[env_ids] = self._sample((len(env_ids), 3), self.angvel_lim)

    def _sample(self, size, scale):
        return scale * 2 * (torch.rand(size, device=self.device) - 0.5)

    def _sample_within_distance(self, size, scale, kWP, dist):
        """next wp is spawned near prev wp"""
        pos = self._sample((size, kWP, 3), scale)
        for i in range(kWP - 1):
            pos[:, i + 1] = pos[:, i] + self._sample((size, 3), dist)

        pos[..., 2] += self.pos_lim
        pos[..., 2] = torch.where(
            pos[..., 2] <= self.min_z, pos[..., 2] + self.min_z, pos[..., 2]
        )
        return pos

    def reset(self, env_ids):
        self.idx[env_ids] = 0
        self.sample(env_ids)
