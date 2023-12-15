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

        self.err_sum, self.prev_err = 0.0, 0.0

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
