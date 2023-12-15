import math
import sys

import torch
from common.torch_jit_utils import *
from env.base.vec_env import VecEnv
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

from .base.vec_env import VecEnv


class BlimpRand(VecEnv):
    def __init__(self, cfg):
        # task-specific parameters
        self.num_obs = 31  #
        self.num_act = 4  #
        self.reset_dist = 10.0  # when to reset

        self.spawn_height = cfg["blimp"].get("spawn_height", 15)

        # domain randomization
        self.num_latent = 25
        self.num_obs += self.num_latent

        super().__init__(cfg=cfg)

        # blimp parameters
        # smoothning factor for fan thrusts
        self.ema_smooth = cfg["blimp"].get("ema_smooth", 0.3)
        self.drag_bodies = torch.tensor(
            cfg["blimp"]["drag_body_idxs"], device=self.sim_device
        ).to(torch.long)
        self.body_areas = torch.tensor(cfg["blimp"]["areas"], device=self.sim_device)
        self.drag_coefs = torch.tensor(
            cfg["blimp"]["drag_coef"], device=self.sim_device
        )
        self.blimp_mass = cfg["blimp"]["mass"]
        self.body_torque_coeff = torch.tensor(
            [0.47, 1.29, 270.0, 5.0, 2.5], device=self.sim_device
        )  # [coef, p, BL4, balance torque, fin torque coef]

        # wind
        self.wind_dirs = torch.tensor(cfg["aero"]["wind_dirs"], device=self.sim_device)
        self.wind_mag = cfg["aero"]["wind_mag"]
        self.wind_std = cfg["aero"]["wind_std"]

        # randomized env latents
        self.range_effort_thrust = [4.0, 6.0]
        self.range_effort_botthrust = [1.5, 4.5]

        self.range_ema_smooth = [self.ema_smooth * 0.5, self.ema_smooth * 1.5]
        self.range_body_areas0 = [self.body_areas[0] * 0.8, self.body_areas[0] * 1.2]
        self.range_body_areas1 = [self.body_areas[1] * 0.8, self.body_areas[1] * 1.2]
        self.range_body_areas2 = [self.body_areas[2] * 0.8, self.body_areas[2] * 1.2]
        self.range_drag_coefs0 = [self.drag_coefs[0] * 0.8, self.drag_coefs[0] * 1.2]
        self.range_drag_coefs1 = [self.drag_coefs[1] * 0.8, self.drag_coefs[1] * 1.2]
        self.range_wind_mag = [self.wind_mag * 0.5, self.wind_mag * 2.0]
        self.range_wind_std = [self.wind_std * 0.8, self.wind_std * 1.2]
        self.range_blimp_mass = [self.blimp_mass * 0.8, self.blimp_mass * 1.2]
        self.range_body_torque_coeff = [
            self.body_torque_coeff * 0.8,
            self.body_torque_coeff * 1.2,
        ]
        self.k_effort_thrust = torch.zeros(self.num_envs, device=self.sim_device)
        self.k_effort_botthrust = torch.zeros(self.num_envs, device=self.sim_device)

        self.k_ema_smooth = torch.zeros(self.num_envs, device=self.sim_device)
        self.k_body_areas = torch.zeros((self.num_envs, 9, 3), device=self.sim_device)
        self.k_drag_coefs = torch.zeros((self.num_envs, 9, 3), device=self.sim_device)
        self.k_wind_mean = torch.zeros((self.num_envs, 3), device=self.sim_device)
        self.k_wind_std = torch.zeros((self.num_envs, 3), device=self.sim_device)
        self.k_wind = torch.zeros((self.num_envs, 3), device=self.sim_device)
        self.k_blimp_mass = torch.zeros(self.num_envs, device=self.sim_device)
        self.k_bouyancy = torch.zeros(self.num_envs, device=self.sim_device)
        self.k_body_torque_coeff = torch.zeros(
            (self.num_envs, 5), device=self.sim_device
        )

        self.randomize_latent()

        # task specific buffers
        # the goal position and rotation are not randomized, instead
        # the spawning pos and rot of the blimp is, since the network gets relative
        # values, these are functionally the same
        self.goal_pos = torch.tile(
            torch.tensor(
                cfg["task"]["target_pos"], device=self.sim_device, dtype=torch.float32
            ),
            (self.num_envs, 1),
        )
        self.goal_pos[..., 2] = self.spawn_height

        self.goal_rot = torch.tile(
            torch.tensor(
                cfg["task"]["target_ang"], device=self.sim_device, dtype=torch.float32
            ),
            (self.num_envs, 1),
        )
        self.goal_rot[..., 0:2] = 0.0  # set roll and pitch zero

        self.goal_lvel = torch.tile(
            torch.tensor(
                cfg["task"]["target_vel"], device=self.sim_device, dtype=torch.float32
            ),
            (self.num_envs, 1),
        )
        self.goal_lvel[..., 2] = 0.0  # set vz zero

        self.goal_avel = torch.tile(
            torch.tensor(
                cfg["task"]["target_angvel"],
                device=self.sim_device,
                dtype=torch.float32,
            ),
            (self.num_envs, 1),
        )
        self.goal_avel[..., 0:2] = 0.0  # set wx, wy zero

        # initialise envs and state tensors
        self.envs = self.create_envs()

        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.gym.prepare_sim(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.rb_states = gymtorch.wrap_tensor(rb_state_tensor)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self.rb_pos = self.rb_states[:, 0:3].view(self.num_envs, self.num_bodies, 3)
        self.rb_rot = self.rb_states[:, 3:7].view(self.num_envs, self.num_bodies, 4)
        self.rb_lvels = self.rb_states[:, 7:10].view(self.num_envs, self.num_bodies, 3)
        self.rb_avels = self.rb_states[:, 10:13].view(self.num_envs, self.num_bodies, 3)

        self.prev_actions = torch.zeros(
            (self.num_envs, self.num_act),
            device=self.sim_device,
            dtype=torch.float,
        )

        # storing tensors for visualisations
        self.actions_tensor = torch.zeros(
            (self.num_envs, self.num_bodies, 3),
            device=self.sim_device,
            dtype=torch.float,
        )
        self.actions_tensor_prev = torch.zeros(
            (self.num_envs, self.num_bodies, 3),
            device=self.sim_device,
            dtype=torch.float,
        )
        self.torques_tensor = torch.zeros(
            (self.num_envs, self.num_bodies, 3),
            device=self.sim_device,
            dtype=torch.float,
        )

        # step simulation to initialise tensor buffers
        self.reset()

    def create_envs(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.restitution = 1
        plane_params.static_friction = 0
        plane_params.dynamic_friction = 0
        self.gym.add_ground(self.sim, plane_params)

        # define environment space (for visualisation)
        spacing = self.goal_lim
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        # add blimp asset
        asset_root = "assets"
        asset_file = "blimp/urdf/blimp.urdf"
        asset_options = gymapi.AssetOptions()
        blimp_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        self.num_dof = self.gym.get_asset_dof_count(blimp_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(blimp_asset)

        # define blimp pose
        pose = gymapi.Transform()
        pose.p.z = self.spawn_height  # generate the blimp 1m from the ground
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # generate environments
        self.blimp_handles = []
        envs = []
        print(f"Creating {self.num_envs} environments.")
        for i in range(self.num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # add blimp here in each environment
            blimp_handle = self.gym.create_actor(
                env_ptr, blimp_asset, pose, "pointmass", i, 1, 0
            )

            dof_props = self.gym.get_actor_dof_properties(env_ptr, blimp_handle)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"].fill(1000.0)
            dof_props["damping"].fill(500.0)

            self.gym.set_actor_dof_properties(env_ptr, blimp_handle, dof_props)

            envs.append(env_ptr)
            self.blimp_handles.append(blimp_handle)

        return envs

    def get_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        # refreshes the rb state tensor with new values
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        roll, pitch, yaw = get_euler_xyz(self.rb_rot[env_ids, 0, :])

        # maps rpy from -pi to pi
        pitch = torch.where(pitch > torch.pi, pitch - 2 * torch.pi, pitch)
        yaw = torch.where(yaw > torch.pi, yaw - 2 * torch.pi, yaw)
        roll = torch.where(roll > torch.pi, roll - 2 * torch.pi, roll)

        # robot angle
        self.obs_buf[env_ids, 0] = roll
        self.obs_buf[env_ids, 1] = pitch
        self.obs_buf[env_ids, 2] = yaw

        # relative angles
        self.obs_buf[env_ids, 3] = roll - self.goal_rot[env_ids, 0]
        self.obs_buf[env_ids, 4] = pitch - self.goal_rot[env_ids, 1]
        self.obs_buf[env_ids, 5] = yaw - self.goal_rot[env_ids, 2]

        # robot sin cos angle
        sin_y = torch.sin(pitch)
        cos_y = torch.cos(pitch)

        sin_z = torch.sin(yaw)
        cos_z = torch.cos(yaw)

        self.obs_buf[env_ids, 6] = sin_y
        self.obs_buf[env_ids, 7] = cos_y

        self.obs_buf[env_ids, 8] = sin_z
        self.obs_buf[env_ids, 9] = cos_z

        # robot z
        self.obs_buf[env_ids, 10] = self.rb_pos[env_ids, 0, 2]

        # relative pos
        rel_pos = self.rb_pos[env_ids, 0] - self.goal_pos[env_ids]
        self.obs_buf[env_ids, 11] = rel_pos[:, 0]
        self.obs_buf[env_ids, 12] = rel_pos[:, 1]
        self.obs_buf[env_ids, 13] = rel_pos[:, 2]

        # relative yaw to goal position
        desired_yaw = torch.arctan2(rel_pos[:, 1], rel_pos[:, 0]) - torch.pi
        ang_to_goal = desired_yaw - yaw
        ang_to_goal = torch.where(
            ang_to_goal > torch.pi, ang_to_goal - 2 * torch.pi, ang_to_goal
        )
        ang_to_goal = torch.where(
            ang_to_goal < -torch.pi, ang_to_goal + 2 * torch.pi, ang_to_goal
        )
        self.obs_buf[env_ids, 14] = ang_to_goal

        # robot vel
        vel = self.rb_lvels[env_ids, 0]

        xv, yv, zv = globalToLocalRot(roll, pitch, yaw, vel[:, 0], vel[:, 1], vel[:, 2])
        self.obs_buf[env_ids, 15] = xv
        self.obs_buf[env_ids, 16] = yv
        self.obs_buf[env_ids, 17] = zv

        # relative vel
        vel = self.rb_lvels[env_ids, 0] - self.goal_lvel[env_ids]

        xv, yv, zv = globalToLocalRot(roll, pitch, yaw, vel[:, 0], vel[:, 1], vel[:, 2])
        self.obs_buf[env_ids, 18] = xv
        self.obs_buf[env_ids, 19] = yv
        self.obs_buf[env_ids, 20] = zv

        # robot angular velocities
        ang_vel = self.rb_avels[env_ids, 0]

        xw, yw, zw = globalToLocalRot(
            roll,
            pitch,
            yaw,
            ang_vel[:, 0],
            ang_vel[:, 1],
            ang_vel[:, 2],
        )
        self.obs_buf[env_ids, 21] = xw
        self.obs_buf[env_ids, 22] = yw
        self.obs_buf[env_ids, 23] = zw

        # rel angular velocities
        ang_vel = self.rb_avels[env_ids, 0] - self.goal_avel[env_ids]

        xw, yw, zw = globalToLocalRot(
            roll,
            pitch,
            yaw,
            ang_vel[:, 0],
            ang_vel[:, 1],
            ang_vel[:, 2],
        )
        self.obs_buf[env_ids, 24] = xw
        self.obs_buf[env_ids, 25] = yw
        self.obs_buf[env_ids, 26] = zw

        # rudder
        # self.obs_buf[env_ids, 27] = self.dof_pos[env_ids, 1]

        # elevator
        # self.obs_buf[env_ids, 28] = self.dof_pos[env_ids, 2]

        # thrust vectoring angle
        # self.obs_buf[env_ids, 29] = self.dof_pos[env_ids, 0]

        # thrust
        # self.obs_buf[env_ids, 30] = self.actions_tensor_prev[env_ids, 3, 2]

        # previous actions
        self.obs_buf[env_ids, 27] = self.prev_actions[env_ids, 0]
        self.obs_buf[env_ids, 28] = self.prev_actions[env_ids, 1]
        self.obs_buf[env_ids, 29] = self.prev_actions[env_ids, 2]
        self.obs_buf[env_ids, 30] = self.prev_actions[env_ids, 3]

        # include env_latent to the observation
        d = 31
        self.obs_buf[env_ids, d] = self.k_effort_thrust[env_ids]
        d += 1
        self.obs_buf[env_ids, d] = self.k_effort_botthrust[env_ids]
        d += 1
        self.obs_buf[env_ids, d] = self.k_ema_smooth[env_ids]

        d += 1
        self.obs_buf[env_ids, d] = self.k_body_areas[env_ids, 0, 0]
        d += 1
        self.obs_buf[env_ids, d] = self.k_body_areas[env_ids, 0, 1]
        d += 1
        self.obs_buf[env_ids, d] = self.k_body_areas[env_ids, 0, 2]
        d += 1
        self.obs_buf[env_ids, d] = self.k_body_areas[env_ids, 1, 1]
        d += 1
        self.obs_buf[env_ids, d] = self.k_body_areas[env_ids, 2, 1]

        d += 1
        self.obs_buf[env_ids, d] = self.k_drag_coefs[env_ids, 0, 0]
        d += 1
        self.obs_buf[env_ids, d] = self.k_drag_coefs[env_ids, 0, 1]
        d += 1
        self.obs_buf[env_ids, d] = self.k_drag_coefs[env_ids, 0, 2]
        d += 1
        self.obs_buf[env_ids, d] = self.k_drag_coefs[env_ids, 1, 1]

        d += 1
        self.obs_buf[env_ids, d] = self.k_body_torque_coeff[env_ids, 0]
        d += 1
        self.obs_buf[env_ids, d] = self.k_body_torque_coeff[env_ids, 1]
        d += 1
        self.obs_buf[env_ids, d] = self.k_body_torque_coeff[env_ids, 2]
        d += 1
        self.obs_buf[env_ids, d] = self.k_body_torque_coeff[env_ids, 3]
        d += 1
        self.obs_buf[env_ids, d] = self.k_body_torque_coeff[env_ids, 4]

        d += 1
        self.obs_buf[env_ids, d] = self.k_wind[env_ids, 0]
        d += 1
        self.obs_buf[env_ids, d] = self.k_wind[env_ids, 1]
        d += 1
        self.obs_buf[env_ids, d] = self.k_wind[env_ids, 2]

        d += 1
        self.obs_buf[env_ids, d] = self.k_wind_mean[env_ids, 0]
        d += 1
        self.obs_buf[env_ids, d] = self.k_wind_mean[env_ids, 1]
        d += 1
        self.obs_buf[env_ids, d] = self.k_wind_mean[env_ids, 2]

        d += 1
        self.obs_buf[env_ids, d] = self.k_blimp_mass[env_ids]
        d += 1
        self.obs_buf[env_ids, d] = self.k_bouyancy[env_ids]

    def randomize_latent(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        def sample_from_range(some_range, dim=1):
            a, b = some_range
            return (
                torch.rand(size=(len(env_ids), dim), device=self.sim_device) * (b - a)
                + a
            )

        # randomize effort
        self.k_effort_thrust[env_ids] = sample_from_range(self.range_effort_thrust)[0]
        self.k_effort_botthrust[env_ids] = sample_from_range(
            self.range_effort_botthrust
        )[0]
        self.k_ema_smooth[env_ids] = sample_from_range(self.range_ema_smooth)[0]

        # randomize dragbody
        a = sample_from_range(self.range_body_areas0, 3)
        b = sample_from_range(self.range_body_areas1, 3)
        c = sample_from_range(self.range_body_areas2, 3)
        self.k_body_areas[env_ids, 0] = a
        self.k_body_areas[env_ids, 1] = b
        self.k_body_areas[env_ids, 3] = b
        self.k_body_areas[env_ids, 5] = b
        self.k_body_areas[env_ids, 7] = b
        self.k_body_areas[env_ids, 2] = c
        self.k_body_areas[env_ids, 4] = c
        self.k_body_areas[env_ids, 6] = c
        self.k_body_areas[env_ids, 8] = c

        self.k_drag_coefs[env_ids, 0] = sample_from_range(self.range_drag_coefs0, 3) * 2
        self.k_drag_coefs[env_ids, 1:] = (
            sample_from_range(self.range_drag_coefs1, 3)[:, None] * 2
        )

        self.k_body_torque_coeff[env_ids] = sample_from_range(
            self.range_body_torque_coeff, 5
        )

        # randomize wind
        k_wind_mag = sample_from_range(self.range_wind_mag, 3)
        k_wind_dir = (
            2 * torch.rand((len(env_ids), 3), device=self.sim_device) * self.wind_dirs
            - self.wind_dirs
        )
        self.k_wind_mean[env_ids] = k_wind_mag * k_wind_dir
        self.k_wind_std[env_ids] = sample_from_range(self.range_wind_std, 3)

        # randomize bouyancy
        self.k_blimp_mass[env_ids] = sample_from_range(self.range_blimp_mass)[0]
        self.k_bouyancy[env_ids] = torch.normal(
            mean=-self.sim_params.gravity.z * (self.k_blimp_mass[env_ids] - 0.5),
            std=0.3,
        )

    def get_reward(self):
        # retrieve environment observations from buffer
        x = self.obs_buf[:, 11]
        y = self.obs_buf[:, 12]
        z = self.obs_buf[:, 13]

        z_abs = self.obs_buf[:, 10]

        (
            self.reward_buf[:],
            self.reset_buf[:],
            self.return_buf[:],
            self.truncated_buf[:],
        ) = compute_point_reward(
            x,
            y,
            z,
            z_abs,
            self.reset_dist,
            self.reset_buf,
            self.progress_buf,
            self.return_buf,
            self.truncated_buf,
            self.max_episode_length,
        )

    def reset(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) == 0:
            return

        # randomise initial positions and velocities
        positions = (
            2
            * (
                torch.rand((len(env_ids), self.num_bodies, 3), device=self.sim_device)
                - 0.5
            )
            * self.goal_lim
        )
        positions[:, :, 2] += self.spawn_height

        velocities = (
            2
            * (
                torch.rand((len(env_ids), self.num_bodies, 6), device=self.sim_device)
                - 0.5
            )
            * self.vel_lim
        )
        velocities[..., 2:5] = 0

        rotations = 2 * (
            (torch.rand((len(env_ids), 4), device=self.sim_device) - 0.5) * math.pi
        )
        rotations[..., 0:2] = 0

        self.goal_rot[env_ids, 2] = rotations[:, 3]

        # domain randomization
        self.randomize_latent(env_ids)

        if self.train and self.rand_vel_targets:
            self.goal_lvel[env_ids, :] = (
                2
                * (torch.rand((len(env_ids), 3), device=self.sim_device) - 0.5)
                * self.goal_vel_lim
            )

            self.goal_avel[env_ids, :] = (
                2
                * (torch.rand((len(env_ids), 3), device=self.sim_device) - 0.5)
                * self.goal_vel_lim
            )

        # set random pos, rot, vels
        self.rb_pos[env_ids, :] = positions[:]

        self.rb_rot[env_ids, 0, :] = quat_from_euler_xyz(
            rotations[:, 0], rotations[:, 1], rotations[:, 2]
        )

        if self.init_vels:
            self.rb_lvels[env_ids, :] = velocities[..., 0:3]
            self.rb_avels[env_ids, :] = velocities[..., 3:6]

        # selectively reset the environments
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.rb_states[::15].contiguous()),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        # clear relevant buffers
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.return_buf[env_ids] = 0
        self.truncated_buf[env_ids] = 0

        self.prev_actions[env_ids] = torch.zeros(
            (len(env_ids), self.num_act),
            device=self.sim_device,
            dtype=torch.float,
        )

        # refresh new observation after reset
        self.get_obs()

    def step(self, actions):
        actions = actions.to(self.sim_device).reshape((self.num_envs, self.num_act))
        actions = torch.clamp(actions, -1.0, 1.0)  # [thrust, yaw, stick, pitch]
        self.prev_actions = actions

        # zeroing out any prev action
        self.actions_tensor[:] = 0.0
        self.torques_tensor[:] = 0.0

        self.actions_tensor[:, 3, 2] = (
            self.k_effort_thrust * (actions[:, 0] + 1) / 2
        )  # propeller
        self.actions_tensor[:, 4, 2] = (
            self.k_effort_thrust * (actions[:, 0] + 1) / 2
        )  # propeller
        self.actions_tensor[:, 7, 1] = (
            self.k_effort_botthrust * actions[:, 1]
        )  # bot propeller

        # EMA smoothing thrusts
        self.actions_tensor[:, [3, 4, 7], :] = self.actions_tensor[
            :, [3, 4, 7], :
        ] * self.ema_smooth + self.actions_tensor_prev[:, [3, 4, 7], :] * (
            1 - self.ema_smooth
        )
        self.actions_tensor_prev[:, [3, 4, 7], :] = self.actions_tensor[:, [3, 4, 7], :]

        # buoyancy
        self.actions_tensor[:] = simulate_buoyancy(
            self.rb_rot, self.k_bouyancy, self.actions_tensor
        )

        # randomize wind
        self.k_wind = torch.normal(mean=self.k_wind_mean, std=self.k_wind_std)

        self.actions_tensor[:], self.torques_tensor[:] = simulate_aerodynamics(
            rb_rot=self.rb_rot,
            rb_avels=self.rb_avels,
            rb_lvels=self.rb_lvels,
            wind=self.k_wind,
            drag_bodies=self.drag_bodies,
            body_areas=self.k_body_areas,
            drag_coefs=self.k_drag_coefs,
            torques_tensor=self.torques_tensor,
            actions_tensor=self.actions_tensor,
            body_torque_coeff=self.k_body_torque_coeff,
        )

        dof_targets = torch.zeros((self.num_envs, 5), device=self.sim_device)
        dof_targets[:, 0] = torch.pi / 2 * actions[:, 2]  # stick
        dof_targets[:, 1] = 0.5 * actions[:, 1]  # bot fin
        dof_targets[:, 4] = -0.5 * actions[:, 1]  # top fin
        dof_targets[:, 2] = 0.5 * actions[:, 3]  # left fin
        dof_targets[:, 3] = -0.5 * actions[:, 3]  # right fin

        # unwrap tensors
        dof_targets = gymtorch.unwrap_tensor(dof_targets)
        forces = gymtorch.unwrap_tensor(self.actions_tensor)
        torques = gymtorch.unwrap_tensor(self.torques_tensor)

        # apply actions
        self.gym.apply_rigid_body_force_tensors(
            self.sim, forces, torques, gymapi.LOCAL_SPACE
        )
        self.gym.set_dof_position_target_tensor(self.sim, dof_targets)

        # simulate and render
        self.simulate()
        if not self.headless:
            self.render()

        # reset environments if required
        self.progress_buf += 1

        self.get_obs()
        self.get_reward()

    def _add_goal_lines(self, num_lines, line_colors, line_vertices, envs):
        num_lines += 2
        line_colors += [[0, 0, 0], [0, 0, 0]]
        for i in range(envs):
            vertices = [
                [self.goal_pos[i, 0].item(), self.goal_pos[i, 1].item(), 0],
                [
                    self.goal_pos[i, 0].item(),
                    self.goal_pos[i, 1].item(),
                    self.goal_pos[i, 2].item(),
                ],
                [
                    self.goal_pos[i, 0].item(),
                    self.goal_pos[i, 1].item(),
                    self.goal_pos[i, 2].item(),
                ],
                [
                    self.goal_pos[i, 0].item() + math.cos(self.goal_rot[i, 2].item()),
                    self.goal_pos[i, 1].item() + math.sin(self.goal_rot[i, 2].item()),
                    self.goal_pos[i, 2].item(),
                ],
            ]
            if len(line_vertices) > i:
                line_vertices[i] += vertices
            else:
                line_vertices.append(vertices)
        return num_lines, line_colors, line_vertices

    def _add_thrust_lines(self, num_lines, line_colors, line_vertices, envs):
        num_lines += 3
        line_colors += [[200, 0, 0], [200, 0, 0], [200, 0, 0]]

        s = 1
        idx = 3
        roll, pitch, yaw = get_euler_xyz(self.rb_rot[:, idx, :])
        f1, f2, f3 = localToGlobalRot(
            roll,
            pitch,
            yaw,
            self.actions_tensor[:, idx, 0],
            self.actions_tensor[:, idx, 1],
            self.actions_tensor[:, idx, 2],
        )
        roll, pitch, yaw = get_euler_xyz(self.rb_rot[:, 7, :])
        g1, g2, g3 = localToGlobalRot(
            roll,
            pitch,
            yaw,
            self.actions_tensor[:, 7, 0],
            self.actions_tensor[:, 7, 1],
            self.actions_tensor[:, 7, 2],
        )
        for i in range(envs):
            vertices = []
            for idx in [3, 4]:
                vertices.append(
                    [
                        self.rb_pos[i, idx, 0].item(),
                        self.rb_pos[i, idx, 1].item(),
                        self.rb_pos[i, idx, 2].item(),
                    ]
                )
                vertices.append(
                    [
                        self.rb_pos[i, idx, 0].item() - s * f1[i].item(),
                        self.rb_pos[i, idx, 1].item() - s * f2[i].item(),
                        self.rb_pos[i, idx, 2].item() - s * f3[i].item(),
                    ]
                )
            vertices.append(
                [
                    self.rb_pos[i, 7, 0].item(),
                    self.rb_pos[i, 7, 1].item(),
                    self.rb_pos[i, 7, 2].item(),
                ]
            )
            vertices.append(
                [
                    self.rb_pos[i, 7, 0].item() - s * g1[i].item(),
                    self.rb_pos[i, 7, 1].item() - s * g2[i].item(),
                    self.rb_pos[i, 7, 2].item() - s * g3[i].item(),
                ]
            )
            if len(line_vertices):
                line_vertices[i] += vertices
            else:
                line_vertices.append(vertices)
        return num_lines, line_colors, line_vertices

    def _add_drag_lines(self, num_lines, line_colors, line_vertices, envs):
        num_lines += 1
        line_colors += [[200, 0, 0]]

        s = 50
        idx = 12
        roll, pitch, yaw = get_euler_xyz(self.rb_rot[:, idx, :])
        f1, f2, f3 = localToGlobalRot(
            roll,
            pitch,
            yaw,
            self.actions_tensor[:, idx, 0],
            self.actions_tensor[:, idx, 1],
            self.actions_tensor[:, idx, 2],
        )

        for i in range(envs):
            vertices = [
                [
                    self.rb_pos[i, idx, 0].item(),
                    self.rb_pos[i, idx, 1].item(),
                    self.rb_pos[i, idx, 2].item(),
                ],
                [
                    self.rb_pos[i, idx, 0].item() - s * f1[i].item(),
                    self.rb_pos[i, idx, 1].item() - s * f2[i].item(),
                    self.rb_pos[i, idx, 2].item() - s * f3[i].item(),
                ],
            ]
            if len(line_vertices):
                line_vertices[i] += vertices
            else:
                line_vertices.append(vertices)
        return num_lines, line_colors, line_vertices

    def _generate_lines(self):
        num_lines = 0
        # line_colors = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [200, 0, 0], [0, 200, 0], [0, 0, 200]]
        line_colors = []
        line_vertices = []

        num_lines, line_colors, line_vertices = self._add_goal_lines(
            num_lines, line_colors, line_vertices, self.num_envs
        )
        num_lines, line_colors, line_vertices = self._add_thrust_lines(
            num_lines, line_colors, line_vertices, self.num_envs
        )
        # num_lines, line_colors, line_vertices = self._add_drag_lines(
        #     num_lines, line_colors, line_vertices, self.num_envs
        # )

        return line_vertices, line_colors, num_lines


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def simulate_buoyancy(rb_rot, bouyancy, actions_tensor):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    roll, pitch, yaw = get_euler_xyz(rb_rot[:, 0, :])
    xa, ya, za = globalToLocalRot(
        roll,
        pitch,
        yaw,
        torch.zeros_like(bouyancy),
        torch.zeros_like(bouyancy),
        bouyancy,
    )
    actions_tensor[:, 0, 0] = xa
    actions_tensor[:, 0, 1] = ya
    actions_tensor[:, 0, 2] = za

    return actions_tensor


@torch.jit.script
def simulate_aerodynamics(
    rb_rot,
    rb_avels,
    rb_lvels,
    wind,
    drag_bodies,
    body_areas,
    drag_coefs,
    torques_tensor,
    actions_tensor,
    body_torque_coeff,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    coef = body_torque_coeff[:, 0]  # coef = 0.47
    p = body_torque_coeff[:, 1]  # p = 1.29
    BL4 = body_torque_coeff[:, 2]  # BL4 = 270
    balance_torque = body_torque_coeff[:, 3]  # balance_torque = 5
    fin_torque_coeff = body_torque_coeff[:, 4]  # fin_torque_coeff = 2.5

    D = (1 / 64) * p * coef * BL4
    r, p, y = get_euler_xyz(rb_rot[:, 0, :])
    a, b, c = globalToLocalRot(
        r,
        p,
        y,
        rb_avels[:, 0, 0],
        rb_avels[:, 0, 1],
        rb_avels[:, 0, 2],
    )
    torques_tensor[:, 0, 1] = -D * b * torch.abs(b)
    torques_tensor[:, 0, 2] = -D * c * torch.abs(c)

    r, p, y = get_euler_xyz_multi(rb_rot[:, drag_bodies, :])
    a, b, c = globalToLocalRot(
        r,
        p,
        y,
        rb_lvels[:, drag_bodies, 0] + wind[:, 0:1],
        rb_lvels[:, drag_bodies, 1] + wind[:, 1:2],
        rb_lvels[:, drag_bodies, 2] + wind[:, 2:3],
    )
    # area = body_areas[i]
    aerodynamic_force0 = -drag_coefs[..., 0] * body_areas[..., 0] * a * torch.abs(a)
    aerodynamic_force1 = -drag_coefs[..., 1] * body_areas[..., 1] * b * torch.abs(b)
    aerodynamic_force2 = -drag_coefs[..., 2] * body_areas[..., 2] * c * torch.abs(c)

    actions_tensor[:, drag_bodies, 0] += aerodynamic_force0
    actions_tensor[:, drag_bodies, 1] += aerodynamic_force1
    actions_tensor[:, drag_bodies, 2] += aerodynamic_force2

    # balance pitch torque
    # torques_tensor[:, drag_bodies[0], 1] += balance_torque

    # pitch torque
    torques_tensor[:, drag_bodies[0], 1] += fin_torque_coeff * (
        aerodynamic_force1[:, 4] - aerodynamic_force1[:, 6]
    )
    # yaw torque
    torques_tensor[:, drag_bodies[0], 2] += fin_torque_coeff * (
        -aerodynamic_force1[:, 2] + aerodynamic_force1[:, 8]
    )
    return actions_tensor, torques_tensor


@torch.jit.script
def compute_point_reward(
    x_pos,
    y_pos,
    z_pos,
    z_abs,
    reset_dist,
    reset_buf,
    progress_buf,
    return_buf,
    truncated_buf,
    max_episode_length,
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor,Tensor, Tensor,float) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    sqr_dist = (x_pos) ** 2 + (y_pos) ** 2 + (z_pos) ** 2

    prox_x_rew_gauss = (torch.exp(-0.01 * sqr_dist) + torch.exp(-0.4 * sqr_dist)) / 2
    # prox_x_rew = torch.where(sqr_dist > 2**2, prox_x_rew_gauss, 1)

    reward = prox_x_rew_gauss

    # adjust reward for reset agents
    reward = torch.where(z_abs < 2, torch.ones_like(reward) * -10.0, reward)
    # reward = torch.where(torch.abs(x_pos) > reset_dist, torch.ones_like(reward) * -200.0, reward)
    # reward = torch.where(torch.abs(y_pos) > reset_dist, torch.ones_like(reward) * -200.0, reward)
    # reward = torch.where(torch.abs(wz) > 45, torch.ones_like(reward) * -200.0, reward)
    # reward = torch.where(x_action < 0, reward - 0.1, reward)
    # reward = torch.where((torch.abs(x_pos) < 0.1) & (torch.abs(y_pos) < 0.1), reward + 1, reward)

    return_buf += reward

    reset = torch.where(
        torch.abs(sqr_dist) > 9000, torch.ones_like(reset_buf), reset_buf
    )
    reset = torch.where(z_abs < 2, torch.ones_like(reset_buf), reset)

    # reset = torch.where(torch.abs(wz) > 70, torch.ones_like(reset_buf), reset)
    # reset = torch.where(torch.abs(wy) > 70, torch.ones_like(reset_buf), reset)
    # reset = torch.where(torch.abs(wx) > 70, torch.ones_like(reset_buf), reset)
    # reset = torch.where(
    #     progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset
    # )

    truncated_buf = torch.where(
        progress_buf >= max_episode_length - 1,
        torch.ones_like(reset_buf),
        truncated_buf,
    )

    return reward, reset, return_buf, truncated_buf
