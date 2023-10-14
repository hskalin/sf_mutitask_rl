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
        self.num_obs = 21  # pole_angle + pole_vel + cart_vel + cart_pos
        self.num_act = 5  # force applied on the pole (-1 to 1)
        self.reset_dist = 10.0  # when to reset
        self.max_push_effort = 5.0  # the range of force applied to the blimp
        self.max_episode_length = 2000  # maximum episode length

        self.ball_height = 8

        super().__init__(cfg=cfg)

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
        self.goal_pos[..., 2] = self.ball_height

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

        # storing tensors for visualisations
        self.actions_tensor = torch.zeros(
            (self.num_envs, self.num_bodies, 3),
            device=self.sim_device,
            dtype=torch.float,
        )

        self.smooth = 0.45
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
        pose.p.z = self.ball_height  # generate the blimp 1m from the ground
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
        positions = (
            2
            * (
                torch.rand((len(env_ids), self.num_bodies, 3), device=self.sim_device)
                - 0.5
            )
            * self.goal_lim
        )
        positions[:, :, 2] = self.ball_height

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

        self.actions_tensor[:] = 0.0
        self.torques_tensor[:] = 0.0

        bouyancy = torch.ones(self.num_envs, device="cuda:0") * 9.8 * 2.32

        roll, pitch, yaw = get_euler_xyz(self.rb_rot[:, 0, :])
        xa, ya, za = globalToLocalRot(
            roll,
            pitch,
            yaw,
            torch.zeros_like(bouyancy),
            torch.zeros_like(bouyancy),
            bouyancy,
        )
        self.actions_tensor[:, 0, 0] = xa
        self.actions_tensor[:, 0, 1] = ya
        self.actions_tensor[:, 0, 2] = za

        self.actions_tensor[:, 3, 2] = 1.5 * (actions[:, 0] + 1) / 2
        self.actions_tensor[:, 4, 2] = 1.5 * (actions[:, 0] + 1) / 2
        self.actions_tensor[:, 7, 1] = 1.5 * actions[:, 1]

        # EMA smoothing thrusts
        self.actions_tensor[:,[3,4,7],:] =  self.actions_tensor[:,[3,4,7],:]*self.smooth + \
                                            self.actions_tensor_prev[:,[3,4,7],:]*(1-self.smooth)
        self.actions_tensor_prev[:,[3,4,7],:] = self.actions_tensor[:,[3,4,7],:]

        # self.torques_tensor[:, 0, 0] = 0 #TK_X + actions[:, 1]
        coef = 1
        p = 1.29
        BL4 = 1.659
        D = (1 / 64) * p * coef * BL4
        r, p, y = get_euler_xyz(self.rb_rot[:, 0, :])
        a, b, c = globalToLocalRot(
            r,
            p,
            y,
            self.rb_avels[:, 0, 0],
            self.rb_avels[:, 0, 1],
            self.rb_avels[:, 0, 2],
        )
        self.torques_tensor[:, 0, 1] = -D * b * torch.abs(b)
        self.torques_tensor[:, 0, 2] = -D * c * torch.abs(c)

        idxs = [0, 7, 8, 9, 10, 11, 12, 13, 14]
        areas = [
            [2.01, 3.84, 3.84],
            [0, 0.25, 0],
            [0, 0.11, 0],
            [0, 0.25, 0],
            [0, 0.11, 0],
            [0, 0.25, 0],
            [0, 0.11, 0],
            [0, 0.25, 0],
            [0, 0.11, 0],
        ]

        wind = torch.normal(mean=torch.tensor([0.9, 0, 0]), std=0.3)

        for i, idx in enumerate(idxs):
            r, p, y = get_euler_xyz(self.rb_rot[:, idx, :])
            a, b, c = globalToLocalRot(
                r,
                p,
                y,
                self.rb_lvels[:, idx, 0]+wind[0],
                self.rb_lvels[:, idx, 1]+wind[1],
                self.rb_lvels[:, idx, 2]+wind[2],
            )
            area = areas[i]
            self.actions_tensor[:, idx, 0] += -coef * area[0] * a * torch.abs(a)
            self.actions_tensor[:, idx, 1] += -coef * area[1] * b * torch.abs(b)
            self.actions_tensor[:, idx, 2] += -coef * area[2] * c * torch.abs(c)

        # unwrap tensors
        forces = gymtorch.unwrap_tensor(self.actions_tensor)
        torques = gymtorch.unwrap_tensor(self.torques_tensor)

        # apply actions
        self.gym.apply_rigid_body_force_tensors(
            self.sim, forces, torques, gymapi.LOCAL_SPACE
        )

        dof_targets = torch.zeros((self.num_envs, 5), device="cuda:0")

        dof_targets[:, 0] = 2.0 * actions[:, 2]
        dof_targets[:, 1] = 0.5 * actions[:, 3]
        dof_targets[:, 4] = -0.5 * actions[:, 3]
        dof_targets[:, 2] = 0.5 * actions[:, 4]
        dof_targets[:, 3] = -0.5 * actions[:, 4]

        dof_targets = gymtorch.unwrap_tensor(dof_targets)
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
                    self.goal_pos[i, 0].item() + math.cos(self.goal_rot[i, 0].item()),
                    self.goal_pos[i, 1].item() + math.sin(self.goal_rot[i, 0].item()),
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

        # idx = 9
        # roll, pitch, yaw = get_euler_xyz(self.rb_rot[:, idx, :])
        # f1, f2, f3 = localToGlobalRot(
        #     roll,
        #     pitch,
        #     yaw,s
        #     self.actions_tensor[:,idx,0],
        #     self.actions_tensor[:,idx,1],
        #     self.actions_tensor[:,idx,2],
        # )
        # a, b, c = globalToLocalRot(
        #         roll,
        #         pitch,
        #         yaw,
        #         self.rb_lvels[:,idx,0],
        #         self.rb_lvels[:,idx,1],
        #         self.rb_lvels[:,idx,2],
        #     )
        # x1, x2, x3 = localToGlobalRot(
        #     roll, pitch, yaw,
        #     a,
        #     torch.zeros(1, device="cuda:0"),
        #     torch.zeros(1, device="cuda:0"),
        # )
        # y1, y2, y3 = localToGlobalRot(
        #     roll, pitch, yaw,
        #     torch.zeros(1, device="cuda:0"),
        #     b,
        #     torch.zeros(1, device="cuda:0"),
        # )
        # z1, z2, z3 = localToGlobalRot(
        #     roll, pitch, yaw,
        #     torch.zeros(1, device="cuda:0"),
        #     torch.zeros(1, device="cuda:0"),
        #     c,
        # )

        # print("vels  ",b)
        # print("forces",self.actions_tensor[:,idx,1])

        # s = 100
        # for i in range(self.num_envs):
        #     vertices = [
        #         [self.goal_pos[i, 0].item(), self.goal_pos[i, 1].item(), 0],
        #         [
        #             self.goal_pos[i, 0].item(),
        #             self.goal_pos[i, 1].item(),
        #             self.goal_pos[i, 2].item(),
        #         ],
        #         [
        #             self.goal_pos[i, 0].item(),
        #             self.goal_pos[i, 1].item(),
        #             self.goal_pos[i, 2].item(),
        #         ],
        #         [
        #             self.goal_pos[i, 0].item() + math.cos(self.goal_rot[i, 0].item()),
        #             self.goal_pos[i, 1].item() + math.sin(self.goal_rot[i, 0].item()),
        #             self.goal_pos[i, 2].item(),
        #         ],
        # [
        #     self.rb_pos[i, idx, 0].item(),
        #     self.rb_pos[i, idx, 1].item(),
        #     self.rb_pos[i, idx, 2].item(),
        # ],
        # [
        #     self.rb_pos[i, idx, 0].item() + s*f1[i].item(),
        #     self.rb_pos[i, idx, 1].item() + s*f2[i].item(),
        #     self.rb_pos[i, idx, 2].item() + s*f3[i].item(),
        # ],
        # [
        #     self.rb_pos[i, idx, 0].item(),
        #     self.rb_pos[i, idx, 1].item(),
        #     self.rb_pos[i, idx, 2].item(),
        # ],
        # [
        #     self.rb_pos[i, idx, 0].item() + s*x1[i].item(),
        #     self.rb_pos[i, idx, 1].item() + s*x2[i].item(),
        #     self.rb_pos[i, idx, 2].item() + s*x3[i].item(),
        # ],
        # [
        #     self.rb_pos[i, idx, 0].item(),
        #     self.rb_pos[i, idx, 1].item(),
        #     self.rb_pos[i, idx, 2].item(),
        # ],
        # [
        #     self.rb_pos[i, idx, 0].item() + s*y1[i].item(),
        #     self.rb_pos[i, idx, 1].item() + s*y2[i].item(),
        #     self.rb_pos[i, idx, 2].item() + s*y3[i].item(),
        # ],
        # [
        #     self.rb_pos[i, idx, 0].item(),
        #     self.rb_pos[i, idx, 1].item(),
        #     self.rb_pos[i, idx, 2].item(),
        # ],
        # [
        #     self.rb_pos[i, idx, 0].item() + s*z1[i].item(),
        #     self.rb_pos[i, idx, 1].item() + s*z2[i].item(),
        #     self.rb_pos[i, idx, 2].item() + s*z3[i].item(),
        # ],
        # ]

        # line_vertices.append(vertices)

        return line_vertices, line_colors, num_lines


#####################################################################
###=========================jit functions=========================###
#####################################################################


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

    prox_x_rew_gauss = torch.exp(-0.05 * sqr_dist)
    # prox_x_rew = torch.where(sqr_dist > 2**2, prox_x_rew_gauss, 1)

    reward = prox_x_rew_gauss

    # adjust reward for reset agents
    reward = torch.where(z_abs < 1, torch.ones_like(reward) * -200.0, reward)
    # reward = torch.where(torch.abs(x_pos) > reset_dist, torch.ones_like(reward) * -200.0, reward)
    # reward = torch.where(torch.abs(y_pos) > reset_dist, torch.ones_like(reward) * -200.0, reward)
    # reward = torch.where(torch.abs(wz) > 45, torch.ones_like(reward) * -200.0, reward)
    # reward = torch.where(x_action < 0, reward - 0.1, reward)
    # reward = torch.where((torch.abs(x_pos) < 0.1) & (torch.abs(y_pos) < 0.1), reward + 1, reward)

    return_buf += reward

    reset = torch.where(
        torch.abs(sqr_dist) > 9000, torch.ones_like(reset_buf), reset_buf
    )

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
