import torch
from scipy.spatial.transform import Rotation as R


class pm_feature:
    def __init__(
        self,
        combination=[True, False, False, False, False, False, False],
        dim=3,
    ) -> None:
        self.envdim = dim

        self._pos_err = combination[0]
        self._pos_norm = combination[1]
        self._vel_err = combination[2]
        self._vel_norm = combination[3]

        self.dim = (
            self.envdim * combination[0]  # pos
            + combination[1]  # pos_norm
            + self.envdim * combination[2]  # vel
            + combination[3]  # vel_norm
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

        return torch.cat(features, 1)


class pointer_feature:
    def __init__(
        self,
        env_cfg,
        combination=[True, True, True, True],
    ) -> None:
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

        self.dim = (
            combination[0]  # pos_norm
            + combination[1]  # vel_norm
            + combination[2]  # ang_norm
            + combination[3]  # angvel_norm
        )

    def extract(self, s):
        features = []

        pos = s[:, 3:5]
        pos_norm = torch.linalg.norm(pos, axis=1, keepdims=True)

        if self._pos_norm:
            features.append(-pos_norm)

        vel = s[:, 5:7]
        if self._vel_norm:
            features.append(-torch.linalg.norm(vel, axis=1, keepdims=True))

        ang = s[:, 0:1]
        if self._ang_norm:
            features.append(-torch.linalg.norm(ang, axis=1, keepdims=True))

        angvel = s[:, 7:8]
        if self._angvel_norm:
            features.append(-torch.linalg.norm(angvel, axis=1, keepdims=True))

        return torch.concatenate(features, 1)
