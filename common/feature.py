import torch
from scipy.spatial.transform import Rotation as R


class pm_feature:
    def __init__(
        self,
        combination=[True, False, False, False, False, False, False],
        success_threshold=(1, 1, 1, 1),
        dim=3,
    ) -> None:
        self.envdim = dim

        self._pos_err = combination[0]
        self._pos_norm = combination[1]
        self._vel_err = combination[2]
        self._vel_norm = combination[3]
        self._ang_err = combination[4]
        self._angvel_err = combination[5]
        self._success = combination[6]

        self._st_pos = success_threshold[0]
        self._st_vel = success_threshold[1]
        self._st_ang = success_threshold[2]
        self._st_angvel = success_threshold[3]

        self.dim = (
            self.envdim * combination[0]  # pos
            + combination[1]  # pos_norm
            + self.envdim * combination[2]  # vel
            + combination[3]  # vel_norm
            + self.envdim * combination[4]  # ang
            + self.envdim * combination[5]  # angvel
            + combination[6]  # suc
        )

    def extract(self, s):
        features = []

        pos = s[:, 0 : self.envdim]
        if self._pos_err:
            features.append(-torch.abs(pos))
        if self._pos_norm:
            features.append(-torch.linalg.norm(pos, axis=1, keepdims=True))

        # vel = s[:, 12:15]
        vel = s[:, self.envdim : 2 * self.envdim]
        if self._vel_err:
            features.append(-torch.abs(vel))
        if self._vel_norm:
            features.append(-torch.linalg.norm(vel, axis=1, keepdims=True))

        if self._success:
            features.append(self.success_position(pos))

        return torch.cat(features, 1)

    def success_position(self, pos):
        dist = torch.linalg.norm(pos, axis=1, keepdims=True)
        suc = torch.zeros_like(dist)
        suc[torch.where(dist < self._st_pos)] = 1.0
        return suc


class pointer_feature:
    def __init__(
        self,
        env_cfg,
        combination=[True, True, True, True, True],
        success_threshold=(1, 1, 1, 1),
    ) -> None:
        self.target_vel = torch.tensor(env_cfg["task"]["target_vel"], device="cuda:0")
        self.envdim = env_cfg["dim"]

        # self._pos_err = combination[0]
        # self._pos_norm = combination[1]
        # self._vel_err = combination[2]
        # self._vel_norm = combination[3]
        # self._ang_err = combination[4]
        # self._angvel_err = combination[5]
        # self._success = combination[6]

        self._pos_norm = combination[0]
        self._vel_norm = combination[1]
        self._ang_norm = combination[2]
        self._angvel_norm = combination[3]
        self._success = combination[4]

        self._st_pos = success_threshold[0]
        self._st_vel = success_threshold[1]
        self._st_ang = success_threshold[2]
        self._st_angvel = success_threshold[3]

        self.dim = (
            combination[0]  # pos_norm
            + combination[1]  # vel_norm
            + combination[2]  # ang_norm
            + combination[3]  # angvel_norm
            + combination[4]  # suc
        )

    def extract(self, s):
        features = []

        pos = s[:, 3:5]
        pos_norm = torch.linalg.norm(pos, axis=1, keepdims=True)

        if self._pos_norm:
            features.append(-pos_norm)

        vel = s[:, 5:7]
        if self._vel_norm:
            features.append(
                -torch.linalg.norm(vel - self.target_vel, axis=1, keepdims=True)
            )

        ang = s[:, 0:1]
        if self._ang_norm:
            features.append(-torch.linalg.norm(ang, axis=1, keepdims=True))

        angvel = s[:, 7:8]
        if self._angvel_norm:
            features.append(-torch.linalg.norm(angvel, axis=1, keepdims=True))

        if self._success:
            features.append(self.success_position(pos_norm))

        return torch.concatenate(features, 1)

    def success_position(self, pos_norm):
        suc = torch.zeros_like(pos_norm)
        suc[torch.where(pos_norm < self._st_pos)] = 1.0
        return suc
