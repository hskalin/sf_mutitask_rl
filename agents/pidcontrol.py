import torch
from torch.optim import Adam

import wandb

from common.agent import IsaacAgent
from common.pid import PIDController
from common.torch_jit_utils import *


class BlimpPositionController(IsaacAgent):
    ctrl_cfg = {
        "yaw": {
            "pid_param": torch.tensor([1.0, 0.01, 0.025]),
            "gain": 1,
        },
        "alt": {
            "pid_param": torch.tensor([1.0, 0.01, 0.5]),
            "gain": 0.2,
        },
        "vel": {
            "pid_param": torch.tensor([0.7, 0.01, 0.5]),
            "gain": 1.0,
        },
    }

    def __init__(self, cfg):
        super().__init__(cfg)

        delta_t = 0.1

        self.yaw_ctrl = PIDController(
            device=self.device,
            delta_t=delta_t,
            **self.ctrl_cfg["yaw"],
        )
        self.alt_ctrl = PIDController(
            device=self.device,
            delta_t=delta_t,
            **self.ctrl_cfg["alt"],
        )
        self.vel_ctrl = PIDController(
            device=self.device,
            delta_t=delta_t,
            **self.ctrl_cfg["vel"],
        )

        self.slice_rb_angle = slice(0, 0 + 3)
        self.slice_goal_angle = slice(3, 3 + 3)
        self.slice_err_posNav = slice(8, 8 + 3)

    # def explore(self, s, w):
    #     err_yaw, err_planar, err_z = self.parse_state(s)

    #     yaw_ctrl = -self.yaw_ctrl.action(err_yaw)
    #     alt_ctrl = self.alt_ctrl.action(err_z)
    #     vel_ctrl = self.vel_ctrl.action(err_planar)
    #     thrust_vec = -1 * torch.ones_like(vel_ctrl)
    #     a = torch.concat([vel_ctrl, yaw_ctrl, thrust_vec, alt_ctrl], dim=1)
    #     # a = torch.tensor([-1, 0, thrust_vec, 0])

    #     return a

    def explore(self, s, w):
        err_heading, err_planar, err_z = self.parse_state(s)

        yaw_ctrl = -self.yaw_ctrl.action(err_heading)
        alt_ctrl = self.alt_ctrl.action(err_z)

        vel_ctrl = torch.where(
            err_z[:, None] <= -3,
            torch.ones_like(alt_ctrl),
            self.vel_ctrl.action(err_planar),
        )
        thrust_vec = torch.where(
            err_z[:, None] <= -3,
            torch.zeros_like(vel_ctrl),
            -1 * torch.ones_like(vel_ctrl),
        )

        a = torch.concat([vel_ctrl, yaw_ctrl, thrust_vec, alt_ctrl], dim=1)
        return a

    def exploit(self, s, w):
        return self.explore(s, w)

    def parse_state(self, s):
        error_posNav = s[:, self.slice_err_posNav]
        robot_angle = s[:, self.slice_rb_angle]

        error_navHeading = check_angle(
            compute_heading(yaw=robot_angle[:, 2], rel_pos=error_posNav)
        )
        err_planar = error_posNav[:, 0:2]
        err_planar = torch.norm(err_planar, dim=1, keepdim=True)
        err_z = error_posNav[:, 2]
        return error_navHeading, err_planar, err_z

    def learn(self):
        pass

    def clear(self):
        self.yaw_ctrl.clear()
        self.alt_ctrl.clear()
        self.vel_ctrl.clear()
