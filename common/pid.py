import torch


class PIDController:
    def __init__(
        self,
        pid_param,
        gain,
        device,
        offset=0.0,
        delta_t=0.01,
        i_from_sensor=False,
        d_from_sensor=False,
    ):
        self.pid_param = torch.tensor(pid_param, device=device)[:, None]
        self.gain = gain
        self.offset = offset
        self.delta_t = delta_t
        self.i_from_sensor = i_from_sensor
        self.d_from_sensor = d_from_sensor

        self.err_sum, self.prev_err = 0, 0

    def action(self, err, err_i=0, err_d=0):
        if err.dim() == 1:
            err = err[:, None]

        if not self.i_from_sensor:
            self.err_sum += err * self.delta_t
            self.err_sum = torch.clip(self.err_sum, -1, 1)
            err_i = self.err_sum

        if not self.d_from_sensor:
            err_d = (err - self.prev_err) / (self.delta_t)
            self.prev_err = err

        pid = torch.concat([err, err_i, err_d], dim=1)
        ctrl = self.gain * (pid @ self.pid_param)
        return ctrl + self.offset

    def clear(self):
        self.err_sum, self.prev_err = 0, 0


class BlimpPositionControl:
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

    def __init__(self, device):
        delta_t = 0.1

        self.yaw_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["yaw"],
        )
        self.alt_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["alt"],
        )
        self.vel_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["vel"],
        )

    def act_on_stack(self, s_stack, k=2):  # [N, S, K]
        self.clear()
        for i in reversed(range(k)):
            a = self.act(s_stack[:, :, -i - 1])
        return a

    def act(self, s):
        err_yaw, err_planar, err_z = self.parse_state(s)

        yaw_ctrl = -self.yaw_ctrl.action(err_yaw)
        alt_ctrl = self.alt_ctrl.action(err_z)
        vel_ctrl = self.vel_ctrl.action(err_planar)
        thrust_vec = -1 * torch.ones_like(vel_ctrl)
        a = torch.concat([vel_ctrl, yaw_ctrl, thrust_vec, alt_ctrl], dim=1)
        return a

    def parse_state(self, s):
        err_planar = s[:, 11:13]
        err_z = s[:, 13]
        err_yaw = s[:, 14]

        err_planar = torch.norm(err_planar, p=2, dim=1, keepdim=True)
        return err_yaw, err_planar, err_z

    def clear(self):
        self.yaw_ctrl.clear()
        self.alt_ctrl.clear()
        self.vel_ctrl.clear()

class BlimpHoverControl:
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

    def __init__(self, device):
        delta_t = 0.1

        self.yaw_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["yaw"],
        )
        self.alt_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["alt"],
        )
        self.vel_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["vel"],
        )

    def act_on_stack(self, s_stack, k=2):  # [N, S, K]
        self.clear()
        for i in reversed(range(k)):
            a = self.act(s_stack[:, :, -i - 1])
        return a

    def act(self, s):
        err_yaw, err_planar, err_z = self.parse_state(s)

        yaw_ctrl = -self.yaw_ctrl.action(err_yaw)
        alt_ctrl = self.alt_ctrl.action(err_z)

        if err_planar <= 10 and err_z <= 5:
            vel_ctrl = self.vel_ctrl.action(torch.abs(err_z))
            thrust_vec = torch.zeros_like(vel_ctrl)
        else:
            vel_ctrl = self.vel_ctrl.action(err_planar)
            thrust_vec = -1*torch.ones_like(vel_ctrl)

        a = torch.concat([vel_ctrl, yaw_ctrl, thrust_vec, alt_ctrl], dim=1)
        return a

    def parse_state(self, s):
        err_planar = s[:, 11:13]
        err_z = s[:, 13]
        err_yaw = s[:, 14]

        err_planar = torch.norm(err_planar, p=2, dim=1, keepdim=True)
        return err_yaw, err_planar, err_z

    def clear(self):
        self.yaw_ctrl.clear()
        self.alt_ctrl.clear()
        self.vel_ctrl.clear()
