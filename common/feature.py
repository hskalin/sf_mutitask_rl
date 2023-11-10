from abc import ABC, abstractmethod

import torch
from scipy.spatial.transform import Rotation as R
import numpy as np


class FeatureAbstract(ABC):
    @abstractmethod
    def extract(self, s):
        """extract state and return hand-crafted features"""
        pass


class PointMassFeature(FeatureAbstract):
    """features
    pos_norm: position norm
    vel_err: velocity error
    vel_norm: velocity norm
    prox: proximity to the goal
    """

    def __init__(
        self,
        env_cfg,
        device,
    ) -> None:
        super().__init__()

        self.env_cfg = env_cfg
        self.feature_cfg = self.env_cfg["feature"]
        self.device = device

        self.use_feature = self.feature_cfg["use_feature"]
        self.verbose = self.feature_cfg.get("verbose", False)

        self.envdim = int(self.env_cfg["dim"])
        self.Kp = torch.tensor(self.env_cfg["goal_lim"], dtype= torch.float64, device=self.device)
        self.Kv = torch.tensor(self.env_cfg["vel_lim"], dtype= torch.float64, device=self.device)
        self.proxThresh = torch.tensor(self.env_cfg["task"]["proximity_threshold"], dtype= torch.float64, device=self.device)
        self.proxRange = 1 / self.compute_gaussDist(
            mu=self.proxThresh**2, sigma=self.Kp, scale=-12.8
        )

        (
            self.use_pos_norm,
            self.use_vel_err,
            self.use_vel_norm,
            self.use_prox,
        ) = self.use_feature
        self.feature_dim = [
            self.use_pos_norm,
            self.envdim * self.use_vel_err,
            self.use_vel_norm,
            self.use_prox,
        ]
        self.dim = int(sum(self.feature_dim))

        self.stateParse_pos = slice(0, self.envdim)
        self.stateParse_vel = slice(self.envdim, 2 * self.envdim)
        self.stateParse_velAbsNorm = slice(2 * self.envdim, 2 * self.envdim + 1)

    def extract(self, s):
        features = []

        pos = s[:, self.stateParse_pos]
        vel = s[:, self.stateParse_vel]
        velAbsNorm = s[:, self.stateParse_velAbsNorm]

        if self.use_pos_norm:
            posSquaredNorm = self.compute_posSquareNorm(pos)
            featurePosNorm = self.compute_featurePosNorm(posSquaredNorm)
            features.append(featurePosNorm)

        if self.use_vel_err:
            featureVel = self.compute_featureVel(vel)
            features.append(featureVel)

        if self.use_vel_norm:
            featureVelNorm = self.compute_featureVelNorm(velAbsNorm)
            features.append(featureVelNorm)

        if self.use_prox:
            posSquaredNorm = self.compute_posSquareNorm(pos)
            proxFeature = self.compute_featureProx(posSquaredNorm)
            features.append(proxFeature)

        return torch.cat(features, 1)

    def compute_posSquareNorm(self, pos):
        return torch.linalg.norm(pos, axis=1, keepdims=True) ** 2

    def compute_featurePosNorm(self, posSquaredNorm, scale=[-7.2, -360]):
        featurePosNorm = 0.5 * (
            self.compute_gaussDist(posSquaredNorm, self.Kp, scale[0])
            + self.compute_gaussDist(posSquaredNorm, self.Kp, scale[1])
        )
        return featurePosNorm

    def compute_featureVel(self, vel, scale=-16):
        return self.compute_gaussDist(vel**2, self.Kv, scale)

    def compute_featureVelNorm(self, velAbsNorm, scale=-16):
        return self.compute_gaussDist(velAbsNorm**2, self.Kv, scale)

    def compute_featureProx(self, posSquaredNorm):
        return torch.where(posSquaredNorm > self.proxThresh**2, self.proxRange, 1)

    def compute_gaussDist(self, mu, sigma, scale):
        return torch.exp(scale * mu / sigma**2)


class PointerFeature(FeatureAbstract):
    def __init__(
        self,
        env_cfg,
        device,
    ) -> None:
        super().__init__()

        self.env_cfg = env_cfg

        self.use_feature = self.feature_cfg["use_feature"]
        
        self.envdim = int(self.env_cfg["dim"])
        self.Kp = torch.tensor(self.env_cfg["goal_lim"], dtype= torch.float64, device=device)
        self.Kv = torch.tensor(self.env_cfg["vel_lim"], dtype= torch.float64, device=device)
        self.proxThresh = torch.tensor(self.env_cfg["task"]["proximity_threshold"], dtype= torch.float64, device=device)
        self.Ka = torch.pi

        (
            self.use_posX,
            self.use_posY,
            self.use_vel_norm,
            self.use_ang_norm,
            self.use_angvel_norm,
        ) = self.use_feature
        self.feature_dim = [
            self.use_posX,
            self.use_posY,
            self.use_vel_norm,
            self.use_ang_norm,
            self.use_angvel_norm
        ]
        self.dim = sum(self.feature_dim)

        self.proxScale = 1 / self.compute_gaussDist(
            mu=self.proxThresh**2, sigma=self.Kp, scale=-25
        )

        self.stateParse_yaw = slice(0, 1)
        self.stateParse_posX = slice(3, 4)
        self.stateParse_posY = slice(4, 5)
        self.stateParse_vel = slice(5, 7)
        self.stateParse_angvel = slice(7, 8)

    def extract(self, s):
        features = []

        errorYaw = s[:, self.stateParse_yaw]
        errorPosX = s[:, self.stateParse_posX]
        errorPosY = s[:, self.stateParse_posY]
        errorVel = s[:, self.stateParse_vel]
        errorAngVel = s[:, self.stateParse_angvel]

        if self.use_posX:
            featureProxX = self.compute_featureProx(errorPosX)
            features.append(featureProxX)

        if self.use_posY:
            featureProxY = self.compute_featureProx(errorPosY)
            features.append(featureProxY)

        if self.use_vel_norm:
            featureVelNorm = self.compute_featureVelNorm(errorVel)
            features.append(featureVelNorm)

        if self.use_ang_norm:
            featureAngNorm = self.compute_featureAngNorm(errorYaw)
            features.append(featureAngNorm)

        if self.use_angvel_norm:
            featureAngVelNorm = self.compute_featureAngVelNorm(errorAngVel)
            features.append(featureAngVelNorm)

        return torch.concatenate(features, 1)

    def compute_featureProx(self, errorPos, scale=-25):
        squaredNorm_pos = errorPos**2
        prox = self.proxScale * self.compute_gaussDist(squaredNorm_pos, self.Kp, scale)
        return torch.where(squaredNorm_pos > self.proxThresh**2, prox, 1)

    def compute_featureVelNorm(self, errorVel, scale=-30):
        SquaredNorm_vel = torch.linalg.norm(errorVel, axis=1, keepdims=True) ** 2
        return self.compute_gaussDist(SquaredNorm_vel, self.Kv, scale)

    def compute_featureAngNorm(self, errorYaw, scale=-50):
        return self.compute_gaussDist(errorYaw**2, self.Ka, scale)

    def compute_featureAngVelNorm(self, errorAngVel, scale=-50):
        return self.compute_gaussDist(errorAngVel**2, self.Kv, scale)

    def compute_gaussDist(self, mu, sigma, scale):
        return torch.exp(scale * mu / sigma**2)



def feature_constructor(env_cfg, device):
    if "pointer" in env_cfg["env_name"].lower():
        return PointerFeature(env_cfg, device)
    elif "pointmass" in env_cfg["env_name"].lower():
        return PointMassFeature(env_cfg, device)
    else:
        print(f'feature not implemented: {env_cfg["env_name"]}')
        return None
