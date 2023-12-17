import isaacgym

import hydra
from omegaconf import DictConfig, OmegaConf
from common.util import (
    omegaconf_to_dict,
    print_dict,
    fix_wandb,
    update_dict,
    AverageMeter,
)

from agents.compose import CompositionAgent
from agents.composeh import RMACompAgent
from agents.sac import SACAgent

import torch
import numpy as np

import wandb

from tkinter import *


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    cfg_dict = omegaconf_to_dict(cfg)

    wandb.init(mode="disabled")
    wandb_dict = fix_wandb(wandb.config)

    # print_dict(wandb_dict)
    update_dict(cfg_dict, wandb_dict)

    
    cfg_dict["buffer"]["n_env"] = cfg_dict["env"]["num_envs"]
    cfg_dict["buffer"]["min_n_experience"] = 0

    # cfg_dict["env"]["episode_max_step"] = int(50 * (512 / cfg_dict["env"]["num_envs"]))
    cfg_dict["env"]["task"]["rand_task"] = False
    cfg_dict["env"]["task"]["rand_vel_targets"] = False
    cfg_dict["env"]["mode"] = "play"
    cfg_dict["env"]["sim"]["headless"] = False
    cfg_dict["env"]["num_envs"] = 1
    print_dict(cfg_dict)

    torch.manual_seed(456)
    np.random.seed(456)

    if "sac" in cfg_dict["agent"]["name"].lower():
        agent = SACAgent(cfg=cfg_dict)
    elif "composition" in cfg_dict["agent"]["name"].lower():
        agent = CompositionAgent(cfg_dict)
    else:
        agent = RMACompAgent(cfg_dict)
        agent.phase=1

    agent.load_torch_model(
        "/home/yutang/rl/sf_mutitask_rl/logs/rmacomp/PointMass2DRand/2023-12-12-05-08-22/model40"
    )

    root = Tk()
    root.title("test")
    root.geometry("300x500")

    weights = agent.w_eval.clone()

    def update_px(val):
        weights[..., 0] = float(val)
        agent.w_eval[:] = weights[:]
        agent.w_eval = agent.w_eval / agent.w_eval.norm(1, 1, keepdim=True)

    def update_py(val):
        weights[..., 1] = float(val)
        agent.w_eval[:] = weights[:]
        agent.w_eval = agent.w_eval / agent.w_eval.norm(1, 1, keepdim=True)

    def update_velnorm(val):
        weights[..., 2] = float(val)
        agent.w_eval[:] = weights[:]
        agent.w_eval = agent.w_eval / agent.w_eval.norm(1, 1, keepdim=True)

    def update_ang(val):
        weights[..., 3] = float(val)
        agent.w_eval[:] = weights[:]
        agent.w_eval = agent.w_eval / agent.w_eval.norm(1, 1, keepdim=True)

    def update_angvel(val):
        weights[..., 4] = float(val)
        agent.w_eval[:] = weights[:]
        agent.w_eval = agent.w_eval / agent.w_eval.norm(1, 1, keepdim=True)

    def update_target_ang(val):
        agent.env.goal_rot[..., 2] = float(val)

    def update_targ_vel(val):
        agent.env.goal_lvel[..., 0] = float(val)

    px_slide = Scale(
        root,
        from_=0,
        to=1,
        digits=3,
        resolution=0.01,
        label="px",
        orient=HORIZONTAL,
        command=update_px,
    )
    px_slide.set(agent.w_eval[0, 0].item())
    px_slide.pack()

    py_slide = Scale(
        root,
        from_=0,
        to=1,
        digits=3,
        resolution=0.01,
        label="py",
        orient=HORIZONTAL,
        command=update_py,
    )
    py_slide.set(agent.w_eval[0, 1].item())
    py_slide.pack()

    vel_slide = Scale(
        root,
        from_=0,
        to=1,
        digits=3,
        resolution=0.01,
        label="vel norm",
        orient=HORIZONTAL,
        command=update_velnorm,
    )
    vel_slide.set(agent.w_eval[0, 2].item())
    vel_slide.pack()

    ang_slide = Scale(
        root,
        from_=0,
        to=1,
        digits=3,
        resolution=0.01,
        label="ang",
        orient=HORIZONTAL,
        command=update_ang,
    )
    ang_slide.set(agent.w_eval[0, 3].item())
    ang_slide.pack()

    angvel_slide = Scale(
        root,
        from_=0,
        to=1,
        digits=3,
        resolution=0.01,
        label="angvel",
        orient=HORIZONTAL,
        command=update_angvel,
    )
    angvel_slide.set(agent.w_eval[0, 4].item())
    angvel_slide.pack()

    targ_vel_slide = Scale(
        root,
        from_=-3.14,
        to=3.14,
        digits=3,
        resolution=0.05,
        label="target vel x",
        orient=HORIZONTAL,
        command=update_targ_vel,
    )
    targ_vel_slide.pack()

    targ_ang_slide = Scale(
        root,
        from_=-3.14,
        to=3.14,
        digits=3,
        resolution=0.05,
        label="target yaw",
        orient=HORIZONTAL,
        command=update_target_ang,
    )
    targ_ang_slide.pack()

    rew = DoubleVar(name="reward")  # instantiate the IntVar variable class
    rew.set(0.0)  # set it to 0 as the initial value

    # the label's textvariable is set to the variable class instance
    Label(root, text="step reward").pack()
    Label(root, textvariable=rew).pack()

    # root.mainloop()

    # while True:
    #     agent.train_episode(root, rew)
    #     if agent.steps > agent.total_timesteps:
    #         break

    avgStepRew = AverageMeter(1, 20).to(agent.device)

    while True:
        s = agent.reset_env()
        for _ in range(5000):
            root.update_idletasks()
            root.update()

            a = agent.act(s, agent.w_eval, "exploit")
            print(agent.w_eval[0])

            agent.env.step(a)
            s_next = agent.env.obs_buf.clone()
            agent.env.reset()

            r = agent.calc_reward(s_next, agent.w_eval)
            s = s_next
            avgStepRew.update(r)
            rew.set(avgStepRew.get_mean())

    wandb.finish()


if __name__ == "__main__":
    launch_rlg_hydra()
