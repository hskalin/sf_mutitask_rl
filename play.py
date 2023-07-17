import isaacgym

import hydra
from omegaconf import DictConfig, OmegaConf
from common.util import omegaconf_to_dict, print_dict, fix_wandb, update_dict

from compose import CompositionAgent
from sac import SACAgent

import torch
import numpy as np

import wandb

from tkinter import *


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    cfg_dict = omegaconf_to_dict(cfg)

    wandb.init(mode="disabled")
    wandb_dict = fix_wandb(wandb.config)

    print_dict(wandb_dict)
    update_dict(cfg_dict, wandb_dict)

    cfg_dict["buffer"]["n_env"] = cfg_dict["env"]["num_envs"]
    # cfg_dict["env"]["episode_max_step"] = int(50 * (512 / cfg_dict["env"]["num_envs"]))
    print_dict(cfg_dict)

    torch.manual_seed(456)
    np.random.seed(456)

    if "sac" in cfg_dict["agent"]["name"].lower():
        agent = SACAgent(cfg=cfg_dict)
    else:
        agent = CompositionAgent(cfg_dict)

    agent.load_torch_model(
        "/home/nilaksh/rl/con_comp/logs/dacgpi/Pointer2D/2023-07-14-17-32-26/model100/"
    )
    # agent.load_torch_model(
    #     "/home/nilaksh/rl/con_comp/logs/dacgpi/Pointer2D/2023-07-14-13-28-23/model100/"
    # )
    agent.env_max_steps = 1000
    agent.eval_episodes = 1
    agent.total_episodes = 20

    # agent.w_eval_navi = torch.tensor([1, 0, 0, 0, 0], device="cuda:0")
    # agent.w_eval_hover = torch.tensor([1, 0, 0.8, 0.2, 0], device="cuda:0")
    # agent.w_eval_init = torch.tile(agent.w_eval_navi, (agent.n_env, 1))  # [N, F]
    # agent.w_eval = agent.w_eval_init.clone().type(torch.float32)
    # print("aaaaaaaaaaaaaaa")
    # print(agent.w_eval)
    # agent.evaluate()

    agent.w_navi = torch.tensor([0, 0, 1, 0, 0], device="cuda:0", dtype=torch.float32)
    agent.w_hover = torch.tensor([0, 0, 1, 0, 0], device="cuda:0", dtype=torch.float32)
    agent.w_init = torch.tile(agent.w_navi, (agent.n_env, 1))  # [N, F]
    agent.w = agent.w_init.clone().type(torch.float32)

    # # print(agent.w)
    # # print(agent.w_navi)
    # # print(agent.w_hover)
    # agent.run()

    # state = agent.env.obs_buf[0]
    # print("aaaaaaaaaaaaaaa")
    # print(state)
    # print(agent.env.trace)

    root = Tk()
    root.title("test")
    root.geometry("300x400")

    def update_pos(val):
        agent.w_navi[0] = float(val)
        agent.w_hover[0] = float(val)
        agent.w_init = torch.tile(agent.w_navi, (agent.n_env, 1))  # [N, F]
        agent.w = agent.w_init.clone().type(torch.float32)

    def update_ang(val):
        agent.w_navi[2] = float(val)
        agent.w_hover[2] = float(val)
        agent.w_init = torch.tile(agent.w_navi, (agent.n_env, 1))  # [N, F]
        agent.w = agent.w_init.clone().type(torch.float32)

    def update_vel(val):
        agent.w_navi[1] = float(val)
        agent.w_hover[1] = float(val)
        agent.w_init = torch.tile(agent.w_navi, (agent.n_env, 1))  # [N, F]
        agent.w = agent.w_init.clone().type(torch.float32)

    def update_suc(val):
        agent.w_navi[4] = float(val)
        agent.w_hover[4] = float(val)
        agent.w_init = torch.tile(agent.w_navi, (agent.n_env, 1))  # [N, F]
        agent.w = agent.w_init.clone().type(torch.float32)

    pos_slide = Scale(
        root,
        from_=0,
        to=1,
        digits=3,
        resolution=0.01,
        label="pos",
        orient=HORIZONTAL,
        command=update_pos,
    )
    pos_slide.pack()

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
    ang_slide.pack()

    vel_slide = Scale(
        root,
        from_=0,
        to=1,
        digits=3,
        resolution=0.01,
        label="vel",
        orient=HORIZONTAL,
        command=update_vel,
    )
    vel_slide.pack()

    suc_slide = Scale(
        root,
        from_=0,
        to=1,
        digits=3,
        resolution=0.01,
        label="suc",
        orient=HORIZONTAL,
        command=update_suc,
    )
    suc_slide.pack()

    rew = DoubleVar(name="reward")  # instantiate the IntVar variable class
    rew.set(0.0)  # set it to 0 as the initial value

    # the label's textvariable is set to the variable class instance
    Label(root, textvariable=rew).pack()

    # root.mainloop()

    while True:
        agent.train_episode(root, rew)
        if agent.steps > agent.total_timesteps:
            break

    wandb.finish()


if __name__ == "__main__":
    launch_rlg_hydra()
