from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from .base.vec_env import VecEnv

from common.torch_jit_utils import *
import sys

import torch
import math


class Blimp(VecEnv):
    def __init__(self, cfg):
        # task-specific parameters
        self.num_obs = 21  #
        self.num_act = 4  # 
        self.reset_dist = 10.0  # when to reset

        self.spawn_height = cfg["task"].get("spawn_height", 15)

        # blimp parameters
        # smoothning factor for fan thrusts
        self.ema_smooth = cfg["blimp"].get("ema_smooth", 0.3)
        self.drag_bodies = torch.tensor(cfg["blimp"]["drag_body_idxs"], device=self.device)
        self.body_areas = torch.tensor(cfg["blimp"]["areas"], device=self.device)
        self.drag_coefs = torch.tensor(cfg["blimp"]["drag_coef"], device=self.device)
        self.blimp_mass = cfg["blimp"]["mass"]

        super().__init__(cfg=cfg)

        # task specific buffers
        # the goal position and rotation are not randomized, instead
        # the spawning pos and rot of the blimp is, since the network gets relative
        # values, these are functionally the same
        self.goal_pos = torch.tile(
            torch.tensor(cfg["task"]["target_pos"], device=self.sim_device, dtype=torch.float32),
            (self.num_envs, 1),)
        self.goal_pos[..., 2] = self.spawn_height

        self.goal_rot = torch.tile(
            torch.tensor(cfg["task"]["target_ang"], device=self.sim_device, dtype=torch.float32),
            (self.num_envs, 1),)
        self.goal_rot[..., 0:2] = 0.0  # set roll and pitch zero

        self.goal_lvel = torch.tile(
            torch.tensor(cfg["task"]["target_vel"], device=self.sim_device, dtype=torch.float32),
            (self.num_envs, 1),)
        self.goal_lvel[..., 2] = 0.0  # set vz zero

        self.goal_avel = torch.tile(
            torch.tensor(cfg["task"]["target_angvel"], device=self.sim_device, dtype=torch.float32,),
            (self.num_envs, 1),)
        self.goal_avel[..., 0:2] = 0.0  # set wx, wy zero

        # wind
        self.wind_dirs = torch.tensor(cfg["aero"]["wind_dirs"], device=self.device)
        self.wind_mag = cfg["aero"]["wind_mag"]
        self.wind_std = cfg["aero"]["wind_std"]
        self.wind_mean = torch.zeros((self.num_envs,3), device=self.device)

        # blimp
        self.bouyancy = torch.zeros(self.num_envs, device=self.device)

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

        # storing tensors for visualisations
        self.actions_tensor = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.sim_device,
            dtype=torch.float,)

        self.actions_tensor_prev = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.sim_device,
            dtype=torch.float,)
        self.torques_tensor = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.sim_device,
            dtype=torch.float,)

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

        # angles
        self.obs_buf[env_ids, 0] = roll - self.goal_rot[env_ids, 0]
        self.obs_buf[env_ids, 1] = pitch - self.goal_rot[env_ids, 1]
        self.obs_buf[env_ids, 2] = yaw - self.goal_rot[env_ids, 2]

        # rotations
        # sine_x = torch.sin(roll)
        # cosine_x = torch.cos(roll)

        sine_y = torch.sin(pitch)
        cosine_y = torch.cos(pitch)

        sine_z = torch.sin(yaw)
        cosine_z = torch.cos(yaw)

        # self.obs_buf[env_ids, 3] = sine_x[env_ids]
        # self.obs_buf[env_ids, 4] = cosine_x[env_ids]

        self.obs_buf[env_ids, 3] = sine_y[env_ids]
        self.obs_buf[env_ids, 4] = cosine_y[env_ids]

        self.obs_buf[env_ids, 5] = sine_z[env_ids]
        self.obs_buf[env_ids, 6] = cosine_z[env_ids]

        # relative xyz pos
        pos = self.rb_pos[env_ids, 0] - self.goal_pos[env_ids]

        self.obs_buf[env_ids, 7] = pos[env_ids, 0]
        self.obs_buf[env_ids, 8] = pos[env_ids, 1]
        self.obs_buf[env_ids, 9] = pos[env_ids, 2]

        # relative xyz vel
        vel = self.rb_lvels[env_ids, 0] - self.goal_lvel[env_ids]

        xv, yv, zv = globalToLocalRot(
            roll, pitch, yaw, vel[env_ids, 0], vel[env_ids, 1], vel[env_ids, 2]
        )
        self.obs_buf[env_ids, 10] = xv
        self.obs_buf[env_ids, 11] = yv
        self.obs_buf[env_ids, 12] = zv

        # angular velocities
        ang_vel = self.rb_avels[env_ids, 0] - self.goal_avel[env_ids]

        xw, yw, zw = globalToLocalRot(
            roll,
            pitch,
            yaw,
            ang_vel[env_ids, 0],
            ang_vel[env_ids, 1],
            ang_vel[env_ids, 2],
        )
        self.obs_buf[env_ids, 13] = xw
        self.obs_buf[env_ids, 14] = yw
        self.obs_buf[env_ids, 15] = zw

        # absolute Z
        self.obs_buf[env_ids, 16] = self.rb_pos[env_ids, 0, 2]

        # rudder
        self.obs_buf[env_ids, 17] = self.dof_pos[env_ids, 1]

        # elevator
        self.obs_buf[env_ids, 18] = self.dof_pos[env_ids, 2]

        # thrust vectoring angle
        self.obs_buf[env_ids, 19] = self.dof_pos[env_ids, 0]

        # thrust
        self.obs_buf[env_ids, 20] = self.actions_tensor[env_ids, 3, 2]
       
    def get_reward(self):
        # retrieve environment observations from buffer
        x = self.obs_buf[:, 7]
        y = self.obs_buf[:, 8]
        z = self.obs_buf[:, 9]

        z_abs = self.obs_buf[:, 16]

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
        positions = (2*(torch.rand((len(env_ids), self.num_bodies, 3), device=self.sim_device) - 0.5
            )*self.goal_lim)
        positions[:, :, 2] += self.spawn_height

        velocities = (2*(torch.rand((len(env_ids), self.num_bodies, 6), device=self.sim_device) - 0.5
            )*self.vel_lim)
        velocities[..., 2:5] = 0

        rotations = 2 * ((torch.rand((len(env_ids), 4), device=self.sim_device) - 0.5) * math.pi)
        rotations[..., 0:2] = 0

        self.goal_rot[env_ids, 2] = rotations[:, 3]

        # randomize wind
        self.wind_mean[env_ids] = self.wind_mag*(2*torch.rand((len(env_ids),3), device=self.device)*self.wind_dirs - self.wind_dirs)

        # randomize bouyancy
        self.bouyancy[env_ids] = torch.normal(-self.sim_params.gravity.z*self.blimp_mass, std=0.3, 
                                        size=(len(env_ids),), device=self.device)

        if self.train and self.rand_vel_targets:
            self.goal_lvel[env_ids, :] = (2*(torch.rand((len(env_ids), 3), device=self.sim_device) - 0.5)*self.goal_vel_lim)

            self.goal_avel[env_ids, :] = (2*(torch.rand((len(env_ids), 3), device=self.sim_device) - 0.5)*self.goal_vel_lim)

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

        # refresh new observation after reset
        self.get_obs()

    def step(self, actions):
        actions = actions.to(self.sim_device).reshape((self.num_envs, self.num_act))
        actions = torch.clamp(actions, -1.0, 1.0)

        # zeroing out any prev action
        self.actions_tensor[:] = 0.0
        self.torques_tensor[:] = 0.0

        self.actions_tensor[:, 3, 2] = 5 * (actions[:, 0] + 1) / 2
        self.actions_tensor[:, 4, 2] = 5 * (actions[:, 0] + 1) / 2
        self.actions_tensor[:, 7, 1] = 2 * actions[:, 1]

        self.actions_tensor[:] = simulate_boyancy(self.rb_rot, self.bouyancy, self.actions_tensor)

        self.actions_tensor[:], self.torques_tensor[:] = simulate_aerodynamics(self.rb_rot, self.rb_avels, self.rb_lvels, 
                                                                               self.wind_mean, self.wind_std, self.drag_bodies, 
                                                                               self.body_areas, self.drag_coefs, self.torques_tensor, self.actions_tensor)

        # EMA smoothing thrusts
        self.actions_tensor[:,[3,4,7],:] =  self.actions_tensor[:,[3,4,7],:]*self.ema_smooth + \
                                            self.actions_tensor_prev[:,[3,4,7],:]*(1-self.ema_smooth)
        self.actions_tensor_prev[:,[3,4,7],:] = self.actions_tensor[:,[3,4,7],:]
        
        dof_targets = torch.zeros((self.num_envs, 5), device=self.device)

        dof_targets[:, 0] =  2.0 * actions[:, 2]
        dof_targets[:, 1] =  0.5 * actions[:, 1]
        dof_targets[:, 4] = -0.5 * actions[:, 1]
        dof_targets[:, 2] =  0.5 * actions[:, 3]
        dof_targets[:, 3] = -0.5 * actions[:, 3]

        # unwrap tensors
        dof_targets = gymtorch.unwrap_tensor(dof_targets)
        forces = gymtorch.unwrap_tensor(self.actions_tensor)
        torques = gymtorch.unwrap_tensor(self.torques_tensor)

        # apply actions
        self.gym.apply_rigid_body_force_tensors(
            self.sim, forces, torques, gymapi.LOCAL_SPACE)
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
        line_colors += [[0, 0, 0]]

        s = 50
        idx = 11
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
        # num_lines,line_colors,line_vertices = self._add_drag_lines(num_lines,line_colors,line_vertices, self.num_envs)

        return line_vertices, line_colors, num_lines


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def simulate_boyancy(rb_rot, bouyancy, actions_tensor):
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
def simulate_aerodynamics(rb_rot, rb_avels, rb_lvels, wind_mean, wind_std, drag_bodies, body_areas, drag_coefs, torques_tensor, actions_tensor):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    coef = 0.47
    p = 1.29
    BL4 = 270
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

    # sample wind 
    wind = torch.normal(mean=wind_mean, std=wind_std)

    r, p, y = get_euler_xyz_multi(rb_rot[:, drag_bodies, :])
    a, b, c = globalToLocalRot(
        r,
        p,
        y,
        rb_lvels[:, drag_bodies, 0] + wind[:,0:1],
        rb_lvels[:, drag_bodies, 1] + wind[:,1:2],
        rb_lvels[:, drag_bodies, 2] + wind[:,2:3],
    )
    # area = body_areas[i]
    actions_tensor[:, drag_bodies, 0] += - drag_coefs[:,0] * body_areas[:, 0] * a * torch.abs(a)
    actions_tensor[:, drag_bodies, 1] += - drag_coefs[:,1] * body_areas[:, 1] * b * torch.abs(b)
    actions_tensor[:, drag_bodies, 2] += - drag_coefs[:,2] * body_areas[:, 2] * c * torch.abs(c)

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

    sqr_dist = (x_pos)**2 + (y_pos)**2 + (z_pos) ** 2

    prox_x_rew_gauss = (torch.exp(-0.01 * sqr_dist) + torch.exp(-0.4 * sqr_dist))/2
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
    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset
    )

    truncated_buf = torch.where(
        progress_buf >= max_episode_length - 1,
        torch.ones_like(reset_buf),
        truncated_buf,
    )

    return reward, reset, return_buf, truncated_buf
