from abc import ABC, abstractmethod

import torch


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

        self.envdim = int(self.env_cfg["feature"]["dim"])
        self.Kp = torch.tensor(
            self.env_cfg["goal_lim"], dtype=torch.float64, device=self.device
        )
        self.Kv = torch.tensor(
            self.env_cfg["vel_lim"], dtype=torch.float64, device=self.device
        )
        self.ProxThresh = torch.tensor(
            self.env_cfg["task"]["proximity_threshold"],
            dtype=torch.float64,
            device=self.device,
        )
        self.proxRange = 1 / self.compute_gaussDist(
            mu=self.ProxThresh**2, sigma=self.Kp, scale=-12.8
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

        self.slice_pos = slice(0, self.envdim)
        self.slice_vel = slice(self.envdim, 2 * self.envdim)
        self.slice_velAbsNorm = slice(2 * self.envdim, 2 * self.envdim + 1)

    def extract(self, s):
        features = []

        pos = s[:, self.slice_pos]
        vel = s[:, self.slice_vel]
        velAbsNorm = s[:, self.slice_velAbsNorm]

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
        return torch.norm(pos, dim=1, keepdim=True) ** 2

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
        return torch.where(posSquaredNorm > self.ProxThresh**2, self.proxRange, 1)

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
        self.feature_cfg = self.env_cfg["feature"]
        self.device = device

        self.use_feature = self.feature_cfg["use_feature"]
        self.verbose = self.feature_cfg.get("verbose", False)

        self.Kp = torch.tensor(
            self.env_cfg["goal_lim"], dtype=torch.float64, device=device
        )
        self.Kv = torch.tensor(
            self.env_cfg["vel_lim"], dtype=torch.float64, device=device
        )
        self.ProxThresh = torch.tensor(
            self.env_cfg["task"]["proximity_threshold"],
            dtype=torch.float64,
            device=device,
        )
        self.Ka = torch.pi

        (
            self.use_posX,
            self.use_posY,
            self.use_vel_norm,
            self.use_ang_norm,
            self.use_angvelNorm,
        ) = self.use_feature

        self.feature_dim = [
            self.use_posX,
            self.use_posY,
            self.use_vel_norm,
            self.use_ang_norm,
            self.use_angvelNorm,
        ]
        self.dim = sum(self.feature_dim)

        self.proxScale = 1 / self.compute_gaussDist(
            mu=self.ProxThresh[None, None], sigma=self.Kp, scale=25
        )

        self.slice_yaw = slice(0, 1)
        self.slice_posX = slice(3, 4)
        self.slice_posY = slice(4, 5)
        self.slice_vel = slice(5, 7)
        self.slice_angvel = slice(7, 8)

    def extract(self, s):
        features = []

        errorYaw = s[:, self.slice_yaw]
        errorPosX = s[:, self.slice_posX]
        errorPosY = s[:, self.slice_posY]
        errorVel = s[:, self.slice_vel]
        errorAngVel = s[:, self.slice_angvel]

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

        if self.use_angvelNorm:
            featureAngVelNorm = self.compute_featureAngVelNorm(errorAngVel)
            features.append(featureAngVelNorm)

        return torch.concatenate(features, 1)

    def compute_featureProx(self, errorPos, scale=25):
        d = torch.norm(errorPos, dim=1, keepdim=True) ** 2
        prox = self.proxScale * torch.exp(scale * -d / self.Kp**2)
        return torch.where(d > self.ProxThresh**2, prox, 1)

    def compute_featureVelNorm(self, errorVel, scale=30):
        return self.compute_gaussDist(errorVel, self.Kv, scale)

    def compute_featureAngNorm(self, errorYaw, scale=50):
        return self.compute_gaussDist(errorYaw, self.Ka, scale)

    def compute_featureAngVelNorm(self, errorAngVel, scale=50):
        return self.compute_gaussDist(errorAngVel, self.Kv, scale)

    def compute_gaussDist(self, mu, sigma, scale):
        mu = torch.norm(mu, dim=1, keepdim=True) ** 2
        return torch.exp(scale * -mu / sigma**2)


class BlimpFeature(FeatureAbstract):
    def __init__(
        self,
        env_cfg,
        device,
    ) -> None:
        super().__init__()

        self.env_cfg = env_cfg
        self.feature_cfg = self.env_cfg["feature"]
        self.device = device

        self.verbose = self.feature_cfg.get("verbose", False)

        self.Kp = torch.tensor(
            self.env_cfg["goal"]["pos_lim"], dtype=torch.float64, device=device
        )
        self.Kv = torch.tensor(
            self.env_cfg["goal"]["vel_lim"], dtype=torch.float64, device=device
        )
        self.ProxThresh = torch.tensor(
            self.env_cfg["task"]["proximity_threshold"],
            dtype=torch.float64,
            device=device,
        )
        self.Ka = torch.pi

        self.dim = 12
        if self.verbose:
            print("[Feature] dim", self.dim)

        self.proxScale = 1 / self.compute_gaussDist(
            mu=self.ProxThresh[None, None], sigma=self.Kp, scale=25
        )

        # robot angle
        self.slice_rbangle = slice(0, 3)
        self.slice_rbRP = slice(0, 2)

        # robot ang vel
        self.slice_rbangvel = slice(21, 24)

        # robot vel
        self.slice_rbv = slice(15, 18)
        self.slice_rbvx = slice(15, 16)
        self.slice_rbvy = slice(16, 17)
        self.slice_rbvz = slice(17, 18)

        # robot thrust
        self.slice_thrust = slice(27, 28)

        # robot actions
        self.slice_prev_act = slice(27, 31)

        # relative angle
        self.slice_err_roll = slice(3, 4)
        self.slice_err_pitch = slice(4, 5)
        self.slice_err_yaw = slice(5, 6)

        # relative position
        self.slice_err_planar = slice(11, 13)
        self.slice_err_z = slice(13, 14)
        self.slice_err_dist = slice(11, 14)

        # relative yaw to goal position
        self.slice_err_yaw_to_goal = slice(14, 15)

        # relative velocity
        self.slice_err_vx = slice(18, 19)
        self.slice_err_vy = slice(19, 20)
        self.slice_err_vz = slice(20, 21)
        self.slice_err_vplanar = slice(18, 20)

        # relative angular velocity
        self.slice_err_p = slice(24, 25)
        self.slice_err_q = slice(25, 26)
        self.slice_err_r = slice(26, 27)
        self.slice_err_angvel = slice(24, 27)

    def extract(self, s):
        features = []

        robot_RP = s[:, self.slice_rbRP]
        robot_angVel = s[:, self.slice_rbangvel]
        robot_v = s[:, self.slice_rbv]
        robot_thrust = s[:, self.slice_thrust]
        # robot_act = s[:, self.slice_prev_act]

        error_yaw = s[:, self.slice_err_yaw]
        error_yaw_to_goal = s[:, self.slice_err_yaw_to_goal]

        error_planar = s[:, self.slice_err_planar]
        error_posZ = s[:, self.slice_err_z]
        error_dist = s[:, self.slice_err_dist]

        error_vx = s[:, self.slice_err_vx]
        error_vy = s[:, self.slice_err_vy]
        error_vz = s[:, self.slice_err_vz]
        # error_vplanar = s[:, self.slice_err_vplanar]

        # error_angVel_p = s[:, self.slice_err_p]
        # error_angVel_q = s[:, self.slice_err_q]
        # error_angVel_r = s[:, self.slice_err_r]
        # error_angVel = s[:, self.slice_err_angvel]

        # planar:
        x = self.compute_featurePosNorm(error_planar)
        features.append(x)

        # posZ:
        x = self.compute_featurePosNorm(error_posZ)
        features.append(x)

        # proxDist:
        x = self.compute_featureProx(error_dist)
        features.append(x)

        # vx:
        x = self.compute_featureVelNorm(error_vx)
        features.append(x)

        # vy:
        x = self.compute_featureVelNorm(error_vy)
        features.append(x)

        # vz:
        x = self.compute_featureVelNorm(error_vz)
        features.append(x)

        # yaw:
        x = self.compute_featureAngNorm(error_yaw)
        features.append(x)

        # yaw_to_goal:
        x = self.compute_featureAngNorm(error_yaw_to_goal)
        features.append(x)

        # regulate_rowandpitch:
        x = self.compute_featureAngNorm(robot_RP)
        features.append(x)

        # regulate_angvel:
        x = self.compute_featureAngVelNorm(robot_angVel)
        features.append(x)

        # regulate robot velocity
        x = self.compute_featureVelNorm(robot_v)
        features.append(x)

        # regulate robot thrust: rescale to [0, 2], similar to angle scale
        x = self.compute_featureAngNorm(robot_thrust + 1)
        features.append(x)

        f = torch.concatenate(features, 1)
        if self.verbose:
            print(
                "[Feature] features [planar, Z, proximity, vx, vy, vz, yaw, yaw2goal, regRP, regPQR, regV, regThrust]"
            )
            print(f)
        return f

    def compute_featurePosNorm(self, x, scale=25):
        return self.compute_gaussDist(x, self.Kp, scale)

    def compute_featureProx(self, x, scale=25):
        d = torch.norm(x, dim=1, keepdim=True) ** 2
        prox = self.proxScale * torch.exp(scale * -d / self.Kp**2)
        return torch.where(d > self.ProxThresh, prox, 1)

    def compute_featureVelNorm(self, x, scale=30):
        return self.compute_gaussDist(x, self.Kv, scale)

    def compute_featureAngNorm(self, x, scale=50):
        return self.compute_gaussDist(x, self.Ka, scale)

    def compute_featureAngVelNorm(self, x, scale=50):
        return self.compute_gaussDist(x, self.Kv, scale)

    def compute_gaussDist(self, mu, sigma, scale):
        mu = torch.norm(mu, dim=1, keepdim=True) ** 2
        return torch.exp(scale * -mu / sigma**2)


class AntFeature(FeatureAbstract):
    """
    features : tbd
    """

    def __init__(self, env_cfg, device) -> None:
        super().__init__()

        self.env_cfg = env_cfg
        self.feature_cfg = self.env_cfg["feature"]
        self.device = device

        self.use_feature = self.feature_cfg["use_feature"]
        self.verbose = self.feature_cfg.get("verbose", False)

        self.envdim = int(self.env_cfg["feature"]["dim"])

        (self.use_pos_x, self.use_pos_y, self.use_alive) = self.use_feature

        self.feature_dim = [self.use_pos_x, self.use_pos_y, self.use_alive]

        self.dim = int(sum(self.feature_dim))

        self.slice_pos_x = slice(0, 1)
        self.slice_pos_y = slice(1, 2)

    def extract(self, s):
        features = []

        if self.use_pos_x:
            features.append(s[:, 0])

        if self.use_pos_y:
            features.append(s[:, 0])

        if self.use_alive:
            features.append(s[:, 0])

        return torch.cat(features, 1)


def feature_constructor(env_cfg, device):
    if "pointer" in env_cfg["env_name"].lower():
        return PointerFeature(env_cfg, device)
    elif "pointmass" in env_cfg["env_name"].lower():
        return PointMassFeature(env_cfg, device)
    elif "ant" in env_cfg["env_name"].lower():
        return AntFeature(env_cfg, device)
    elif "blimp" in env_cfg["env_name"].lower():
        return BlimpFeature(env_cfg, device)
    else:
        print(f'feature not implemented: {env_cfg["env_name"]}')
        return None
