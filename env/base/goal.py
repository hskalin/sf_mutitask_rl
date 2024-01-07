import torch
import math
import numpy as np


def PointsInCircum(r,z,n=8):
    pi = math.pi
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r, z) for x in range(0,n+1)]

def PointsInSquare(r,z,n=8):
    pi = math.pi
    return [(math.cos(2*pi/n*x)*r, math.sin(2*pi/n*x)*r, z) for x in range(0,n+1)]

class FixWayPoints:
    """fixed waypoints"""

    def __init__(
        self,
        device,
        num_envs,
        pos_lim=None,
        trigger_dist=2,
        style="square",
        **kwargs,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.trigger_dist = trigger_dist

        self.pos_lim = pos_lim

        self.Kv = 0.1

        
        self.kWayPt = 8
        if style=="square":
            wps = torch.tensor(
                [[[10, -10, 20], [10, 0, 20], [10, 10, 20], [0, 10, 20], [-10, 10, 20], [-10, 0, 20], [-10, -10, 20], [0, -10, 20]]],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            wps = torch.tensor(
                PointsInCircum(self.pos_lim/2, self.pos_lim, self.kWayPt-1),
                device=self.device,
                dtype=torch.float32,
            )
        self.pos_nav = torch.tile(wps, (self.num_envs, 1, 1))

        self.pos_hov = torch.tile(
            torch.tensor([0, 0, 20], device=self.device, dtype=torch.float32),
            (self.num_envs, 1),
        )

        self.vel = torch.tile(
            torch.tensor([0, 0, 0], device=self.device, dtype=torch.float32),
            (self.num_envs, 1),
        )
        self.velnorm = torch.tile(
            torch.tensor([2.0], device=self.device, dtype=torch.float32),
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

    def sample(self, env_ids):
        pass

    def get_pos_nav(self, idx=None):
        if idx is not None:
            return self.pos_nav[range(self.num_envs), self.check_idx(idx).squeeze()]
        else:
            return self.pos_nav[
                range(self.num_envs), self.check_idx(self.idx).squeeze()
            ]

    def update_state(self, rb_pos):
        """check if robot is close to waypoint"""
        dist = torch.norm(
            rb_pos[:, 0:2] - self.get_pos_nav(self.idx)[:, 0:2],
            p=2,
            dim=1,
            keepdim=True,
        )
        self.idx = torch.where(dist <= self.trigger_dist, self.idx + 1, self.idx)
        trigger = torch.where(dist <= self.trigger_dist, 1.0, 0.0)
        self.idx = self.check_idx(self.idx)

        self.update_vel(rb_pos, Kv=self.Kv)
        return trigger

    def reset(self, env_ids):
        self.idx[env_ids] = 0

    def update_vel(self, rbpos, Kv=0.1):
        cur_pos = self.get_pos_nav(self.idx)
        goal_vec = cur_pos - rbpos
        unit_goal_vec = goal_vec / torch.norm(goal_vec, p=1, keepdim=True)
        self.vel = Kv*(goal_vec + unit_goal_vec)

    def check_idx(self, idx):
        idx = torch.where(idx > self.kWayPt - 1, 0, idx)
        idx = torch.where(idx < 0, self.kWayPt - 1, idx)
        return idx


class RandomWayPoints(FixWayPoints):
    """a random generated goal during training"""

    def __init__(
        self,
        device,
        num_envs,
        init_pos=None,
        init_vel=None,
        init_velnorm=None,
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
        reset_dist=30,
    ) -> None:
        super().__init__(
            device=device,
            num_envs=num_envs,
            pos_lim=pos_lim,
            trigger_dist=trigger_dist,
        )

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

        self.wp_dist = wp_dist
        self.trigger_dist = torch.tensor(trigger_dist).to(self.device)
        self.min_z = min_z
        self.reset_dist = reset_dist

        self.idx = torch.zeros(self.num_envs, 1, device=self.device).to(torch.long)

        assert self.kWayPt > 1, "number of waypoints less than 1"
        wps = torch.tensor(
            PointsInCircum(self.pos_lim/2, self.pos_lim, self.kWayPt-1),
            device=self.device,
            dtype=torch.float32,
        )
        self.pos_nav = torch.tile(wps, (self.num_envs, 1, 1))

        if init_pos is not None:
            self.init_pos = init_pos
            self.pos_hov = torch.tile(
                torch.tensor(init_pos, device=self.device, dtype=torch.float32),
                (self.num_envs, 1),
            )

        if init_vel is not None:
            self.init_vel = init_vel
            self.vel = torch.tile(
                torch.tensor(init_vel, device=self.device, dtype=torch.float32),
                (self.num_envs, 1),
            )

        if init_velnorm is not None:
            self.init_velnorm = init_velnorm
            self.velnorm = torch.tile(
                torch.tensor(init_velnorm, device=self.device, dtype=torch.float32),
                (self.num_envs, 1),
            )

        if init_ang is not None:
            self.init_ang = init_ang
            self.ang = torch.tile(
                torch.tensor(init_ang, device=self.device, dtype=torch.float32),
                (self.num_envs, 1),
            )

        if init_angvel is not None:
            self.init_angvel = init_angvel
            self.angvel = torch.tile(
                torch.tensor(init_angvel, device=self.device, dtype=torch.float32),
                (self.num_envs, 1),
            )

        self.idx = torch.zeros(self.num_envs, 1, device=self.device).to(torch.long)

    def update_state(self, rb_pos):
        """check if robot is close to waypoint"""
        dist = torch.norm(
            rb_pos[:, 0:2] - self.get_pos_nav(self.idx)[:, 0:2],
            p=2,
            dim=1,
            keepdim=True,
        )
        self.idx = torch.where(dist <= self.trigger_dist, self.idx + 1, self.idx)
        trigger = torch.where(dist <= self.trigger_dist, 1.0, 0.0)
        self.idx = self.check_idx(self.idx)

        self.update_vel(rb_pos, Kv=self.Kv)
        return trigger

    def sample(self, env_ids):
        if self.rand_pos:
            self.pos_nav[env_ids] = self._sample_on_distance(
                len(env_ids), self.pos_lim, kWP=self.kWayPt, dist=self.wp_dist
            )

        if self.rand_vel:
            self.velnorm[env_ids] = torch.abs(
                self._sample((len(env_ids), 1), self.vel_lim-1)
            ) + 1 # min 1 [m/s]

        if self.rand_ang:
            self.ang[env_ids] = self._sample((len(env_ids), 3), math.pi)

        if self.rand_angvel:
            self.angvel[env_ids] = self._sample((len(env_ids), 3), self.angvel_lim)

    def _sample(self, size, scale=1):
        return scale * 2 * (torch.rand(size, device=self.device) - 0.5)

    def _sample_on_distance(self, size, scale, kWP, dist):
        """next wp is spawned [dist] from prev wp"""
        # pos = self._sample((size, kWP, 3), scale)
        pos = torch.zeros(size=(size, kWP, 3), device=self.device)
        pos[..., 2] = self.pos_lim
        for i in range(kWP - 1):
            x = self._sample((size, 3))
            x = dist * x / torch.norm(x, dim=1, keepdim=True)
            pos[:, i + 1] = pos[:, i] + x

        pos[..., 2] = torch.where(
            pos[..., 2] <= self.min_z, self.pos_lim, pos[..., 2]
        )
        pos[..., 2] = torch.where(
            pos[..., 2] >= 2*self.pos_lim, self.pos_lim, pos[..., 2]
        )
        return pos


    def reset(self, env_ids):
        self.idx[env_ids] = 0
        self.sample(env_ids)
