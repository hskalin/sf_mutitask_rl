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
        wp_dist=10 / np.sqrt(3),  # [m] generate next wp within range
        trigger_dist=2,  # [m] activate next wp if robot within range
        min_z=5,
        path_vel=True,  # generate velocity command from waypoints
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.kWayPt = kWayPt

        self.path_vel = path_vel if kWayPt >= 2 else False

        self.pos_lim = pos_lim
        self.vel_lim = vel_lim
        self.angvel_lim = angvel_lim

        self.rand_pos = rand_pos
        self.rand_ang = rand_ang
        self.rand_vel = rand_vel
        self.rand_angvel = rand_angvel

        self.wp_dist = wp_dist
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
                (self.num_envs, 1),
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

    def update_state(self, rb_pos, hover_task=None):
        """check if robot is close to waypoint"""
        planar_dist = torch.norm(
            rb_pos[:, 0:2] - self.pos[range(self.num_envs), self.idx.squeeze(), 0:2],
            p=2,
            dim=1,
            keepdim=True,
        )
        self.idx = torch.where(planar_dist <= self.trigger_dist, self.idx + 1, self.idx)
        self.idx = self.check_idx(self.idx)

        if hover_task is not None:
            self.idx = torch.where(hover_task[:, None] == True, 0, self.idx)

        if self.path_vel:
            self.update_vel(rb_pos)

    def sample(self, env_ids):
        if self.rand_pos:
            self.pos[env_ids] = self._sample_on_distance(
                len(env_ids), self.pos_lim, kWP=self.kWayPt, dist=self.wp_dist
            )

        if self.rand_vel:
            self.vel[env_ids] = self._sample((len(env_ids), 3), self.vel_lim)

        if self.rand_ang:
            self.ang[env_ids] = self._sample((len(env_ids), 3), math.pi)

        if self.rand_angvel:
            self.angvel[env_ids] = self._sample((len(env_ids), 3), self.angvel_lim)

    def _sample(self, size, scale=1):
        return scale * 2 * (torch.rand(size, device=self.device) - 0.5)

    def _sample_on_distance(self, size, scale, kWP, dist):
        """next wp is spawned near prev wp"""
        pos = self._sample((size, kWP, 3), scale)
        for i in range(kWP - 1):
            x = self._sample((size, 3))
            x = dist * x / torch.norm(x, dim=1, keepdim=True)
            pos[:, i + 1] = pos[:, i] + x

        pos[..., 2] += self.pos_lim
        pos[..., 2] = torch.where(
            pos[..., 2] <= self.min_z, pos[..., 2] + self.min_z, pos[..., 2]
        )
        return pos

    def reset(self, env_ids):
        self.idx[env_ids] = 0
        self.sample(env_ids)

    def update_vel(self, rbpos, Kv=0.5):
        prev_pos = self.pos[
            range(self.num_envs), self.check_idx(self.idx - 1).squeeze()
        ]  # [N, 3]

        path = self.get_pos() - prev_pos

        k = torch.einsum("ij,ij->i", rbpos - prev_pos, path) / torch.einsum(
            "ij,ij->i", path, path
        )
        k = torch.where(k > 1, 1 - k, k)
        self.vel = Kv * (path + prev_pos + k[:, None] * path - rbpos)

    def check_idx(self, idx):
        idx = torch.where(idx > self.kWayPt - 1, 0, idx)
        idx = torch.where(idx < 0, self.kWayPt - 1, idx)
        return idx


class FixWayPoints:
    """fixed waypoints"""

    def __init__(
        self,
        device,
        num_envs,
        pos_lim=None,
        trigger_dist=2,
        **kwargs,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.trigger_dist = trigger_dist

        self.pos_lim = pos_lim

        self.kWayPt = 4
        wps = torch.tensor(
            [[[20, -20, 15], [20, 20, 15], [-20, 20, 15], [-20, -20, 15]]],
            device=self.device,
            dtype=torch.float32,
        )
        self.pos = torch.tile(wps, (self.num_envs, 1, 1))

        self.vel = torch.tile(
            torch.tensor([5, 0, 0], device=self.device, dtype=torch.float32),
            (self.num_envs, 1),
        )

        self.ang = torch.tile(
            torch.tensor([0, 0, 0], device=self.device, dtype=torch.float32),
            (self.num_envs, 1),
        )

        self.angvel = torch.tile(
            torch.tensor([0, 0, 0], device=self.device, dtype=torch.float32),
            (self.num_envs, 1),
        )

        self.idx = torch.zeros(self.num_envs, 1, device=self.device).to(torch.long)

    def get_pos(self):
        return self.pos[range(self.num_envs), self.idx.squeeze()]

    def update_state(self, rb_pos, hover_task=None):
        """check if robot is close to waypoint"""
        dist = torch.norm(
            rb_pos - self.get_pos(),
            p=2,
            dim=1,
            keepdim=True,
        )
        self.idx = torch.where(dist <= self.trigger_dist, self.idx + 1, self.idx)
        trigger = torch.where(dist <= self.trigger_dist, 1.0, 0.0)
        self.idx = self.check_idx(self.idx)

        if hover_task is not None:
            self.idx = torch.where(hover_task[:, None] == True, 0, self.idx)

        self.update_vel(rb_pos)
        return trigger

    def reset(self, env_ids):
        self.idx[env_ids] = 0

    def update_vel(self, rbpos, Kv=0.5):
        prev_pos = self.pos[
            range(self.num_envs), self.check_idx(self.idx - 1).squeeze()
        ]  # [N, 3]

        path = self.get_pos() - prev_pos

        k = torch.einsum("ij,ij->i", rbpos - prev_pos, path) / torch.einsum(
            "ij,ij->i", path, path
        )
        k = torch.where(k > 1, 1 - k, k)
        self.vel = Kv * (path + prev_pos + k[:, None] * path - rbpos)

    def check_idx(self, idx):
        idx = torch.where(idx > self.kWayPt - 1, 0, idx)
        idx = torch.where(idx < 0, self.kWayPt - 1, idx)
        return idx
