#!/usr/bin/env python

import copy
import math
from tkinter import *
import json
import isaacgym
import hydra
import numpy as np
import rospy
import torch
import wandb
from common.torch_jit_utils import *
from common.util import (
    AverageMeter,
    fix_wandb,
    omegaconf_to_dict,
    print_dict,
    update_dict,
)
from common.pid import BlimpVelocityControl
from env.base.goal import FixWayPoints
from librepilot.msg import LibrepilotActuators
from omegaconf import DictConfig, OmegaConf
from run import get_agent
from sensor_msgs.msg import Imu
from std_srvs.srv import Empty
from uav_msgs.msg import uav_pose
from geometry_msgs.msg import Pose, TwistStamped

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
        self.dt = 0.1

        # init and load agent
        self.agent = get_agent(cfg_dict)
        self.agent.load_torch_model(model_path)

        self.weights = self.agent.task.Eval.W.clone()
        self.weightLabels = cfg_dict["env"]["task"]["taskLabels"]

        self.rew = None
        self.generate_scales()
        self.print_step_reward()

        # init waypints
        self.wp = FixWayPoints(device=self.device, num_envs=1, trigger_dist=5)

        # init buffer
        self.obs_buf = torch.zeros(1, 34, device=self.device)

        self.rb_pos = torch.zeros(1, 3, device=self.device)
        self.rb_lvels = torch.zeros(1, 3, device=self.device)
        self.rb_lacc = torch.zeros(1, 3, device=self.device)
        self.ori_data = torch.zeros(1, 4, device=self.device)
        self.rb_rot = torch.zeros(1, 3, device=self.device)
        self.rb_avels = torch.zeros(1, 3, device=self.device)

        self.prev_actions = torch.zeros((1, 4), device=self.device)
        self.prev_actuator = torch.zeros((1, 3), device=self.device)
        self.ema_smooth = torch.tensor([[2 * self.dt, 3 * self.dt]], device=self.device)

        # init ros node
        rospy.init_node("rl_node")
        self.action_publisher = rospy.Publisher(
            "blimp" + "/GCSACTUATORS", LibrepilotActuators, queue_size=1
        )

        rospy.Subscriber("/blimp/tail/imu", Imu, self._imu_callback)
        rospy.Subscriber("/blimp/ground_speed", TwistStamped, self._vel_callback)
        # rospy.Subscriber("/blimp/tail/pose", uav_pose, self._pose_callback)
        rospy.Subscriber("/blimp/tail/pose", Pose, self._pose_callback)
        self.rate = rospy.Rate(1 / self.dt)

        self.reset_world_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)

        self.ros_cnt = 0

    def _imu_callback(self, msg):
        """imu msg callback

        Args:
            msg ([Imu]): imu sensor raw data
        """
        acc = obj2tensor(msg.linear_acceleration)
        if self.real_exp:
            acc[:, 2] += GRAVITY
        else:
            acc[:, 2] -= GRAVITY

        self.rb_lacc = acc

        if self.dbg_ros:
            self.ros_cnt += 1
            if self.ros_cnt % 100 == 0:
                print(
                    "[ KinematicObservation ] imu_callback: linear_acceleration",
                    self.rb_lacc,
                )

    def _vel_callback(self, msg):
        """imu msg callback

        Args:
            msg ([Imu]): imu sensor raw data
        """
        self.rb_lvels = obj2tensor(msg.twist.linear)

        if self.dbg_ros:
            self.ros_cnt += 1
            if self.ros_cnt % 100 == 0:
                print(
                    "[ KinematicObservation ] vel_callback: velocity",
                    self.rb_lvels,
                )

    def _pose_callback(self, msg):
        """pose msg callback

        Args:
            msg ([uav_pose]): gcs processed sensor data
        """
        self.rb_pos = obj2tensor(msg.position)
        if self.real_exp:  # convert from NED to ENU coordinate
            pos_data = copy.copy(self.rb_pos)
            self.rb_pos[:, 0] = pos_data[:, 1]
            self.rb_pos[:, 1] = pos_data[:, 0]
            self.rb_pos[:, 2] = -pos_data[:, 2]

        # self.rb_lvels = obj2tensor(msg.velocity)
        self.ori_data = obj2tensor(msg.orientation)
        self.rb_rot = torch.concat(get_euler_wxyz(self.ori_data)).unsqueeze(0)

        # self.rb_avels = obj2tensor(msg.angVelocity)

        if self.dbg_ros:
            print(
                "[ KinematicObservation ] pose_callback: position",
                self.rb_pos,
            )
            # print(
            #     "[ KinematicObservation ] pose_callback: velocity",
            #     self.rb_lvels,
            # )
            print(
                "[ KinematicObservation ] pose_callback: orientation",
                self.ori_data,
            )
            print(
                "[ KinematicObservation ] pose_callback: angle",
                self.rb_rot,
            )
            print(
                "[ KinematicObservation ] pose_callback: ang_vel",
                self.rb_avels,
            )

    def reset_world(self):
        """reset gazebo world"""
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as err:
            print("/gazebo/reset_world service call failed", err)

    def pause_sim(self):
        """pause simulation with ros service call"""
        rospy.logdebug("PAUSING START")
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as err:
            print("/gazebo/pause_physics service call failed", err)

        rospy.logdebug("PAUSING FINISH")

    def unpause_sim(self):
        """unpause simulation with ros service call"""
        rospy.logdebug("UNPAUSING START")
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as err:
            print("/gazebo/unpause_physics service call failed", err)

        rospy.logdebug("UNPAUSING FiNISH")

    def play(self):
        def shutdownhook():
            print("shutdown time!")

        # ctrler = BlimpVelocityControl(device=self.device)
        avgStepRew = AverageMeter(1, 20).to(self.agent.device)
        while not rospy.is_shutdown():
            s = self.reset()

            for _ in range(5000):
                self.root.update_idletasks()
                self.root.update()

                a = self.agent.act(s, self.agent.task.Eval, "exploit")
                # a = ctrler.act(s)
                # print("task:", self.agent.task.Eval.W[0])
                # print("action:", a)

                # a: [thrust, yaw, stick, pitch]
                a = self.prepare_action(a)

                # actuator:
                # [m2, lfin, rfin, tfin, bfin, stick, m1, unused, m0, unused, unused, unused]
                actuator = self.map_actuator(a)

                act_msg = LibrepilotActuators()
                act_msg.header.stamp = rospy.Time.now()
                act_msg.data.data = actuator
                self.action_publisher.publish(act_msg)

                self.rate.sleep()

                s_next = self.get_obs()

                r = self.agent.calc_reward(s_next, self.agent.task.Eval.W)
                s = s_next
                avgStepRew.update(r)
                if self.rew:
                    self.rew.set(avgStepRew.get_mean())

                rospy.on_shutdown(shutdownhook)

    def map_actuator(self, a):
        actuator = 1500 * np.ones([12, 1])
        actuator[[6, 8]] = a[0]
        actuator[[0, 3, 4]] = a[1]
        actuator[5] = a[2]
        actuator[[1, 2]] = a[3]
        return actuator

    def prepare_action(self, a):
        a = torch.clip(a, -1, 1)
        self.prev_actions = copy.deepcopy(a)

        a[:, 0] = a[:, 0] * self.ema_smooth[:, 0] + self.prev_actuator[:, 0] * (
            1 - self.ema_smooth[:, 0]
        )
        a[:, 2] = a[:, 2] * self.ema_smooth[:, 1] + self.prev_actuator[:, 1] * (
            1 - self.ema_smooth[:, 1]
        )
        bot_thrust = a[:, 1] * self.ema_smooth[:, 0] + self.prev_actuator[:, 2] * (
            1 - self.ema_smooth[:, 1]
        )

        self.prev_actuator[:, 0] = a[:, 0]
        self.prev_actuator[:, 1] = a[:, 2]
        self.prev_actuator[:, 2] = bot_thrust

        a[:, 0] = torch.clip((a[:, 0] + 1) / 2, 0, 0.7)
        a[:, 1] = -a[:, 1]
        a[:, 3] = -a[:, 3]

        a = lmap(a, [-1, 1], [1000, 2000])
        a = a.squeeze().detach().cpu().numpy()
        return a

    def get_obs(self):
        env_ids = 0
        roll, pitch, yaw = self.rb_rot[env_ids]

        # robot angle
        # self.obs_buf[env_ids, 0] = check_angle(roll)
        self.obs_buf[env_ids, 0] = 0
        self.obs_buf[env_ids, 1] = check_angle(pitch)
        self.obs_buf[env_ids, 2] = check_angle(yaw)

        print("roll", check_angle(roll))
        print("pitch", check_angle(pitch))
        print("yaw", check_angle(yaw))

        # goal angles
        self.obs_buf[env_ids, 3] = check_angle(self.wp.ang[env_ids, 0])
        self.obs_buf[env_ids, 4] = check_angle(self.wp.ang[env_ids, 1])
        self.obs_buf[env_ids, 5] = check_angle(self.wp.ang[env_ids, 2])

        # robot z
        self.obs_buf[env_ids, 6] = self.rb_pos[env_ids, 2]

        print("z", self.rb_pos[env_ids, 2])

        # trigger navigation goal
        trigger = self.wp.update_state(self.rb_pos)
        self.obs_buf[env_ids, 7] = trigger[env_ids, 0]

        print("wp trigger", trigger[env_ids, 0])
        print("wp idx", self.wp.idx)

        # relative pos to navigation goal
        rel_pos = self.rb_pos - self.wp.get_pos_nav()
        self.obs_buf[env_ids, 8] = rel_pos[env_ids, 0]
        self.obs_buf[env_ids, 9] = rel_pos[env_ids, 1]
        self.obs_buf[env_ids, 10] = rel_pos[env_ids, 2]

        print("rel_pos nav", rel_pos)

        # relative pos to hover goal
        rel_pos = self.rb_pos - self.wp.pos_hov
        self.obs_buf[env_ids, 11] = rel_pos[env_ids, 0]
        self.obs_buf[env_ids, 12] = rel_pos[env_ids, 1]
        self.obs_buf[env_ids, 13] = rel_pos[env_ids, 2]

        print("rel_pos hov", rel_pos)

        # robot vel
        self.obs_buf[env_ids, 14] = self.rb_lvels[env_ids, 0]
        self.obs_buf[env_ids, 15] = self.rb_lvels[env_ids, 1]
        self.obs_buf[env_ids, 16] = self.rb_lvels[env_ids, 2]

        print("rb_lvels", self.rb_lvels)

        # goal vel
        self.obs_buf[env_ids, 17] = self.wp.vel[env_ids, 0]
        self.obs_buf[env_ids, 18] = self.wp.vel[env_ids, 1]
        self.obs_buf[env_ids, 19] = self.wp.vel[env_ids, 2]
        self.obs_buf[env_ids, 20] = self.wp.velnorm[env_ids, 0]

        print("wp.vel", self.wp.vel)
        print("wp.velnorm", self.wp.velnorm)

        # robot angular velocities
        self.obs_buf[env_ids, 21] = self.rb_avels[env_ids, 0]
        self.obs_buf[env_ids, 22] = self.rb_avels[env_ids, 1]
        self.obs_buf[env_ids, 23] = self.rb_avels[env_ids, 2]

        # goal ang vel
        self.obs_buf[env_ids, 24] = self.wp.angvel[env_ids, 0]
        self.obs_buf[env_ids, 25] = self.wp.angvel[env_ids, 1]
        self.obs_buf[env_ids, 26] = self.wp.angvel[env_ids, 2]

        # print("wp.angvel", self.wp.angvel)

        # prev actuators
        self.obs_buf[env_ids, 27] = self.prev_actuator[env_ids, 0]  # thrust
        self.obs_buf[env_ids, 28] = self.prev_actuator[env_ids, 1]  # stick
        self.obs_buf[env_ids, 29] = self.prev_actuator[env_ids, 2]  # bot thrust

        print("prev_actuator", self.prev_actuator)

        # previous actions
        self.obs_buf[env_ids, 30] = self.prev_actions[env_ids, 0]
        self.obs_buf[env_ids, 31] = self.prev_actions[env_ids, 1]
        self.obs_buf[env_ids, 32] = self.prev_actions[env_ids, 2]
        self.obs_buf[env_ids, 33] = self.prev_actions[env_ids, 3]

        print("prev_actions", self.prev_actions)

        return self.obs_buf.clone()

    def reset(self):
        self.agent.comp.reset()
        self.agent.prev_traj.clear()

        env_ids = 0

        # sample new waypoint
        self.wp.sample(env_ids)
        self.wp.ang[env_ids, 0:2] = 0
        self.wp.angvel[env_ids, 0:2] = 0

        self.prev_actions[env_ids] = torch.zeros((1, 4), device=self.device)
        self.prev_actuator[env_ids] = torch.zeros((1, 3), device=self.device)

        # reset gazebo world
        self.pause_sim()
        self.reset_world()
        self.unpause_sim()

        # refresh new observation after reset
        return self.get_obs().clone()

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


def modify_cfg(cfg_dict):
    # don't change these
    cfg_dict["agent"]["norm_task_by_sf"] = False

    cfg_dict["buffer"]["n_env"] = cfg_dict["env"]["num_envs"]
    cfg_dict["buffer"]["min_n_experience"] = 0
    cfg_dict["buffer"]["capacity"] = 20

    cfg_dict["env"]["save_model"] = False
    cfg_dict["env"]["task"]["rand_task"] = False
    cfg_dict["env"]["task"]["rand_vel_targets"] = False
    cfg_dict["env"]["mode"] = "play"
    cfg_dict["env"]["sim"]["headless"] = True

    # change these
    cfg_dict["env"]["num_envs"] = 1
    cfg_dict["env"]["goal"]["trigger_dist"] = 10
    cfg_dict["agent"]["phase"] = 4  # phase: [encoder, adaptor, fine-tune, deploy]
    if "aero" in cfg_dict["env"]:
        cfg_dict["env"]["aero"]["wind_mag"] = 0.0
    if "domain_rand" in cfg_dict["env"]["task"]:
        cfg_dict["env"]["task"]["domain_rand"] = False
    cfg_dict["agent"]["exploit_method"] = "sfgpi"

    print_dict(cfg_dict)

    return cfg_dict


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    cfg_dict = omegaconf_to_dict(cfg)

    wandb.init(mode="disabled")
    wandb_dict = fix_wandb(wandb.config)

    model_folder = "/home/yutang/rl/sf_mutitask_rl/logs/rmacompblimp/BlimpRand/2023-12-31-18-00-46/"
    model_checkpoint = "model200"

    cfg_path = model_folder + "/cfg"
    model_path = model_folder + "/" + model_checkpoint + "/"

    cfg_dict = None
    with open(cfg_path) as f:
        cfg_dict = json.load(f)

    update_dict(cfg_dict, wandb_dict)
    cfg_dict = modify_cfg(cfg_dict)

    torch.manual_seed(456)
    np.random.seed(456)

    playob = PlayUI(cfg_dict, model_path)
    playob.play()

    wandb.finish()


if __name__ == "__main__":
    launch_rlg_hydra()
