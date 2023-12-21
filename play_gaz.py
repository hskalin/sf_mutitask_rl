#!/usr/bin/env python

import math
from tkinter import *
import copy
import hydra
import isaacgym
import numpy as np
import rospy
import wandb
from common.util import (
    AverageMeter,
    fix_wandb,
    omegaconf_to_dict,
    print_dict,
    update_dict,
)
from env.base.goal import FixWayPoints
from librepilot.msg import LibrepilotActuators
from omegaconf import DictConfig, OmegaConf
from run import get_agent
from sensor_msgs.msg import Imu
from uav_msgs.msg import uav_pose
import torch

GRAVITY = 9.81


def obj2tensor(
    rosobj,
    attr_list=["w", "x", "y", "z"],
):
    val_list = []
    for attr in attr_list:
        try:
            val_list.append(getattr(rosobj, attr))
        except:
            pass

    return torch.tensor(val_list, device="cuda", dtype=torch.float).unsqueeze(0)


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def lmap(v, x, y) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


class PlayUI:
    def __init__(
        self, cfg_dict, model_path, real_exp=False, dbg_ros=False, device="cuda"
    ) -> None:
        self.root = Tk()
        self.root.title("test")
        self.root.geometry("300x500")

        # init exp params
        self.real_exp = real_exp
        self.dbg_ros = dbg_ros
        self.device = device

        # init and load agent
        self.agent = get_agent(cfg_dict)
        self.agent.load_torch_model(model_path)

        self.weights = self.agent.task.Eval.W.clone()
        self.weightLabels = cfg_dict["env"]["task"]["taskLabels"]

        self.rew = None
        self.generate_scales()
        self.print_step_reward()

        # init buffer
        self.wp = FixWayPoints(device=self.device, num_envs=1, trigger_dist=3)

        self.pos_data = torch.zeros(1, 3, device=self.device)
        self.vel_data = torch.zeros(1, 3, device=self.device)
        self.acc_data = torch.zeros(1, 3, device=self.device)
        self.ori_data = torch.zeros(1, 4, device=self.device)
        self.ang_data = torch.zeros(1, 3, device=self.device)
        self.ang_vel_data = torch.zeros(1, 3, device=self.device)

        # init ros node
        rospy.init_node("rl_node")
        self.action_publisher = rospy.Publisher(
            "blimp" + "/GCSACTUATORS", LibrepilotActuators, queue_size=1
        )

        rospy.Subscriber("/blimp/tail/imu", Imu, self._imu_callback)
        rospy.Subscriber("/blimp/tail/pose", uav_pose, self._pose_callback)
        self.ros_cnt = 0
        self.obs_buf = torch.zeros(1, 30, device=self.device)
        self.prev_act = torch.zeros(1, 4, device=self.device)

    def _imu_callback(self, msg):
        """imu msg callback

        Args:
            msg ([Imu]): imu sensor raw data
        """
        acc = obj2tensor(msg.linear_acceleration)
        if self.real_exp:
            acc[2] += GRAVITY
        else:
            acc[2] -= GRAVITY

        self.acc_data = acc

        if self.dbg_ros:
            self.ros_cnt += 1
            if self.ros_cnt % 100 == 0:
                print(
                    "[ KinematicObservation ] imu_callback: linear_acceleration",
                    self.acc_data,
                )

    def _pose_callback(self, msg):
        """pose msg callback

        Args:
            msg ([uav_pose]): gcs processed sensor data
        """
        self.pos_data = obj2tensor(msg.position)
        if self.real_exp:  # convert from NED to ENU coordinate
            pos_data = copy.copy(self.pos_data)
            self.pos_data[:, 0] = pos_data[:, 1]
            self.pos_data[:, 1] = pos_data[:, 0]
            self.pos_data[:, 2] = -pos_data[:, 2]

        self.vel_data = obj2tensor(msg.velocity)
        self.ori_data = obj2tensor(msg.orientation)
        self.ang_data = euler_from_quaternion(self.ori_data)
        self.ang_vel_data = obj2tensor(msg.angVelocity)

        if self.dbg_ros:
            print(
                "[ KinematicObservation ] pose_callback: position",
                self.pos_data,
            )
            print(
                "[ KinematicObservation ] pose_callback: velocity",
                self.vel_data,
            )
            print(
                "[ KinematicObservation ] pose_callback: orientation",
                self.ori_data,
            )
            print(
                "[ KinematicObservation ] pose_callback: angle",
                self.ang_data,
            )
            print(
                "[ KinematicObservation ] pose_callback: ang_vel",
                self.ang_vel_data,
            )

    def check_angle(self, ang):
        return torch.where(ang > torch.pi, ang - 2 * torch.pi, ang)

    def prepare_state(self):
        env_ids = 0
        # self.pos_data
        # self.vel_data
        # self.acc_data

        # self.ori_data
        # self.ang_data,
        # self.ang_vel_data

        roll, pitch, yaw = self.ang_data[env_ids]

        # maps rpy from -pi to pi
        pitch = self.check_angle(pitch)
        roll = self.check_angle(roll)
        yaw = self.check_angle(yaw)

        # robot angle
        self.obs_buf[env_ids, 0] = roll
        self.obs_buf[env_ids, 1] = pitch
        self.obs_buf[env_ids, 2] = yaw

        # relative angles
        rel_roll = roll - self.wp.ang[env_ids, 0]
        rel_pitch = pitch - self.wp.ang[env_ids, 1]
        rel_yaw = yaw - self.wp.ang[env_ids, 2]

        rel_roll = self.check_angle(rel_roll)
        rel_pitch = self.check_angle(rel_pitch)
        rel_yaw = self.check_angle(rel_yaw)

        self.obs_buf[env_ids, 3] = rel_roll
        self.obs_buf[env_ids, 4] = rel_pitch
        self.obs_buf[env_ids, 5] = rel_yaw

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
        self.obs_buf[env_ids, 10] = self.pos_data[env_ids, 0, 2]

        # relative pos
        self.wp.update_state(self.rb_pos[:, 0], self.hover_task)
        rel_pos = self.rb_pos[env_ids, 0] - self.wp.get_pos()[env_ids]
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
        vel = self.rb_lvels[env_ids, 0] - self.wp.vel[env_ids]

        xv, yv, zv = globalToLocalRot(roll, pitch, yaw, vel[:, 0], vel[:, 1], vel[:, 2])
        self.obs_buf[env_ids, 18] = xv
        self.obs_buf[env_ids, 19] = yv
        self.obs_buf[env_ids, 20] = zv

        # robot angular velocities
        ang_vel = self.rb_avels[env_ids, 0]

        self.obs_buf[env_ids, 21] = ang_vel[:, 0]
        self.obs_buf[env_ids, 22] = ang_vel[:, 1]
        self.obs_buf[env_ids, 23] = ang_vel[:, 2]

        # rel angular velocities
        ang_vel = self.rb_avels[env_ids, 0] - self.wp.angvel[env_ids]

        self.obs_buf[env_ids, 24] = ang_vel[:, 0]
        self.obs_buf[env_ids, 25] = ang_vel[:, 1]
        self.obs_buf[env_ids, 26] = ang_vel[:, 2]

        # previous actions
        self.obs_buf[env_ids, 27] = self.prev_actions[env_ids, 0]
        self.obs_buf[env_ids, 28] = self.prev_actions[env_ids, 1]
        self.obs_buf[env_ids, 29] = self.prev_actions[env_ids, 2]
        self.obs_buf[env_ids, 30] = self.prev_actions[env_ids, 3]

    def weight_update_function(self, dimension):
        def update_val(val):
            self.weights[..., dimension] = float(val)
            self.agent.task.Eval.W[:] = self.weights[:]
            self.agent.task.Eval.W = (
                self.agent.task.Eval.W / self.agent.task.Eval.W.norm(1, 1, keepdim=True)
            )

        return update_val

    def target_update_function(self, dimension):
        def update_val(val):
            self.agent.env.goal_pos[..., dimension] = float(val)

        return update_val

    def add_scale(self, dimension, gen_func, label, range=(0, 1), type="weight"):
        scale = Scale(
            self.root,
            from_=range[0],
            to=range[1],
            digits=3,
            resolution=0.01,
            label=label,
            orient=HORIZONTAL,
            command=gen_func(dimension),
        )
        if type == "weight":
            scale.set(self.agent.task.Eval.W[0, dimension].item())
        scale.pack()

    def generate_scales(self):
        for i, label in enumerate(self.weightLabels):
            self.add_scale(
                dimension=i, gen_func=self.weight_update_function, label=label
            )

        self.add_scale(
            dimension=0,
            gen_func=self.target_update_function,
            label="target pos",
            range=(-2, 2),
            type="target",
        )

    def print_step_reward(self):
        self.rew = DoubleVar(name="reward")  # instantiate the IntVar variable class
        self.rew.set(0.0)  # set it to 0 as the initial value

        # the label's textvariable is set to the variable class instance
        Label(self.root, text="step reward").pack()
        Label(self.root, textvariable=self.rew).pack()

    def _debug_ui(self):
        # only runs UI loop without inference
        while True:
            self.root.update_idletasks()
            self.root.update()

            print(self.agent.task.Eval.W[0])

    def play(self):
        def shutdownhook():
            print("shutdown time!")

        avgStepRew = AverageMeter(1, 20).to(self.agent.device)
        while not rospy.is_shutdown():
            s = self.agent.reset_env()
            for _ in range(5000):
                self.root.update_idletasks()
                self.root.update()

                a = self.agent.act(s, self.agent.task.Eval, "exploit")
                print(self.agent.task.Eval.W[0])

                # a: [thrust, yaw, stick, pitch]
                a = a.squeeze()
                a = torch.clip(a, -1, 1)
                a = lmap(a, [-1, 1], [1000, 2000])

                # actuator:
                # [m2, lfin, rfin, tfin, bfin, stick, m1, unused, m0, unused, unused, unused]
                actuator = 1500 * np.ones([12, 1])
                actuator[6, 8] = a[0]
                actuator[0, 3, 4] = a[1]
                actuator[5] = a[2]
                actuator[1, 2] = a[2]

                act_msg = LibrepilotActuators()
                act_msg.header.stamp = rospy.Time.now()
                act_msg.data.data = actuator
                self.action_publisher.publish(act_msg)

                # ==== TODO: obtain gaz obs to isaac format ====#
                s_next = self.agent.env.obs_buf.clone()

                r = self.agent.calc_reward(s_next, self.agent.task.Eval.W)
                s = s_next
                avgStepRew.update(r)
                if self.rew:
                    self.rew.set(avgStepRew.get_mean())

                rospy.on_shutdown(shutdownhook)


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    cfg_dict = omegaconf_to_dict(cfg)

    wandb.init(mode="disabled")
    wandb_dict = fix_wandb(wandb.config)

    # print_dict(wandb_dict)
    update_dict(cfg_dict, wandb_dict)
    cfg_dict["agent"]["norm_task_by_sf"] = True
    cfg_dict["agent"]["phase"] = 2

    cfg_dict["buffer"]["n_env"] = cfg_dict["env"]["num_envs"]
    cfg_dict["buffer"]["min_n_experience"] = 0

    # cfg_dict["env"]["episode_max_step"] = int(50 * (512 / cfg_dict["env"]["num_envs"]))
    cfg_dict["env"]["task"]["rand_task"] = False
    cfg_dict["env"]["task"]["rand_vel_targets"] = False
    cfg_dict["env"]["mode"] = "play"
    cfg_dict["env"]["sim"]["headless"] = False
    cfg_dict["env"]["num_envs"] = 1

    cfg_dict["env"]["aero"]["wind_mag"] = 0
    cfg_dict["env"]["task"]["domain_rand"] = False
    print_dict(cfg_dict)

    torch.manual_seed(456)
    np.random.seed(456)

    model_path = "/home/yutang/rl/sf_mutitask_rl/logs/rmacompblimp/BlimpRand/2023-12-21-01-50-43/model90"

    playob = PlayUI(cfg_dict, model_path)
    playob.play()

    wandb.finish()


if __name__ == "__main__":
    launch_rlg_hydra()
