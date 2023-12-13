import torch
from torch.optim import Adam

import wandb
from common.agent import IsaacAgent


class PIDController:
    def __init__(
        self,
        pid_param=torch.tensor([1.0, 0.2, 0.05]),
        gain=1.0,
        offset=0.0,
        delta_t=0.01,
        i_from_sensor=False,
        d_from_sensor=False,
    ):
        self.pid_param = pid_param
        self.gain = gain
        self.offset = offset
        self.delta_t = delta_t
        self.i_from_sensor = i_from_sensor
        self.d_from_sensor = d_from_sensor

        self.err_sum, self.prev_err = 0.0, 0.0
        self.windup = 0.0

    def action(self, err, err_i=0, err_d=0):
        if not self.i_from_sensor:
            self.err_sum += err * self.delta_t
            self.err_sum = torch.clip(self.err_sum, -1, 1)
            err_i = self.err_sum * (1 - self.windup)

        if not self.d_from_sensor:
            err_d = (err - self.prev_err) / (self.delta_t)
            self.prev_err = err

        ctrl = self.gain * torch.dot(self.pid_param, torch.tensor([err, err_i, err_d]))
        return ctrl + self.offset

    def clear(self):
        self.err_sum, self.prev_err = 0, 0
        self.windup = 0.0


class BlimpPIDControl(IsaacAgent):
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
            delta_t=delta_t,
            **self.ctrl_cfg["yaw"],
        )
        self.alt_ctrl = PIDController(
            delta_t=delta_t,
            **self.ctrl_cfg["alt"],
        )
        self.vel_ctrl = PIDController(
            delta_t=delta_t,
            **self.ctrl_cfg["vel"],
        )

    def explore(self, s, w):
        err_yaw, err_planar, err_z = self.parse_state(s)

        yaw_ctrl = -self.yaw_ctrl.action(err_yaw)
        alt_ctrl = self.alt_ctrl.action(err_z)
        vel_ctrl = self.vel_ctrl.action(err_planar)
        a = torch.tensor([vel_ctrl, yaw_ctrl, -1, alt_ctrl])

        return a

    def exploit(self, s, w):
        return self.explore(s, w)

    def parse_state(self, s):
        err_planar = s[:, 11:13]
        err_z = s[:, 13]
        err_yaw = s[:, 14]

        err_planar = torch.norm(err_planar, p=2, dim=1, keepdim=True)
        return err_yaw, err_planar, err_z

    def learn(self):
        pass
