from abc import ABC, abstractmethod
import torch
import itertools


class TaskAbstraction(ABC):
    @abstractmethod
    def define_task(self):
        """define task weight"""
        pass


class SmartTask(TaskAbstraction):
    def __init__(self, env_cfg, device) -> None:
        super().__init__()

        self.env_cfg = env_cfg
        self.task_cfg = self.env_cfg["task"]
        self.device = device

        self.verbose = self.task_cfg.get("verbose", False)
        self.n_env = self.env_cfg["num_envs"]
        self.use_feature = self.env_cfg["feature"]["use_feature"]

        env_dim = self.env_cfg["dim"]
        init_wTrain = self.task_cfg["task_w"]
        self.init_wTrain = self.define_task(self.use_feature, env_dim, init_wTrain)
        self.init_wTrain = torch.tensor(
            self.init_wTrain, device="cuda:0", dtype=torch.float32
        )
        self.dim = int(self.init_wTrain.shape[0])

        init_wEval = self.task_cfg["task_w_eval"]
        self.init_wEval = self.define_task(self.use_feature, env_dim, init_wEval)
        self.init_wEval = torch.tensor(
            self.init_wEval, device="cuda:0", dtype=torch.float32
        )

        # /////////////////////////////////////////
        self.randMethod = self.task_cfg.get("task_w_randType", "permute")
        self.intervalWeightRand = self.task_cfg.get("intervalWeightRand", 2)
        # self.minWeightVecs = 5

        self.idx_envTask = self.uniformRand_taskSet(self.randMethod)
        self.task_ratio = torch.ones(len(self.get_taskSet(self.randMethod)), device=self.device)

        if self.task_cfg["rand_weights"]:
            self.wEval = self.uniformRand_taskSet("achievable") # [N, F]
            self.wEval = self.normalize_task(self.wEval)
            self.randTrainTask() 
        else:
            self.wEval = torch.tile(self.init_wEval, (self.n_env, 1))  # [N, F]
            self.wEval = self.normalize_task(self.wEval)
            self.wTrain = torch.tile(self.init_wTrain, (self.n_env, 1))
            self.wTrain = self.normalize_task(self.wTrain)

        if self.verbose:
            print("[Task] evaluation task:\n", self.wEval)

    def define_task(self):
        """define task weight"""
        NotImplementedError

    def randTask(self, episodes):
        if (
            ((episodes - 1) % self.intervalWeightRand == 0)
            and (self.env_cfg["mode"] == "train")
            and (self.task_cfg["rand_weights"])
        ):
            self.randTrainTask()

        if self.verbose:
            print("[Task] w_train[0]: ", self.wTrain[0])

    def uniformRand_taskSet(self, method):
        taskLst = self.get_taskSet(method)
        taskTensor = torch.tensor(taskLst, device=self.device, dtype=torch.float32)
        idx_envTask = self.uniform_sample_tasks(taskLst)
        return taskTensor[idx_envTask]

    def get_taskSet(self, randMethod):
        if randMethod == "permute":
            taskLst = list(itertools.product([0, 1], repeat=self.dim))
            taskLst.pop(0)  # remove all zero vector
        elif randMethod == "identity":
            taskLst = [
                [1 if i == j else 0 for j in range(self.dim)] for i in range(self.dim)
            ]
        elif randMethod == "achievable":
            taskLst = self.task_cfg["task_wa"]
        elif randMethod == "single":
            taskLst = self.task_cfg["task_ws"]
        else:
            raise ValueError(f"{randMethod} no implemented")
        return taskLst

    def uniform_sample_tasks(self, taskLst):
        n_task=len(taskLst)
        task_ratio = torch.ones(n_task, device=self.device)
        idx_envTask = self.sample_task(task_ratio)
        return idx_envTask

    def sample_task(self, ratio):
        """sample tasks based on their ratio"""
        return torch.multinomial(ratio, self.n_env, replacement=True)

    def normalize_task(self, w):
        return w / w.norm(1, 1, keepdim=True)

    def adapt_task(self, episode_r):
        if self.task_cfg["task_w_randAdaptive"]:
            self.task_ratio = self.task_ratio.index_add(
                dim=0, index=self.idx_envTask, source=episode_r.float()
            )

    def randTrainTask(self):
        if self.randMethod == "single":
            wTrain = self.uniformRand_taskSet(self.randMethod)

        elif self.randMethod == "permute" or "identity" or "achievable":
            taskLst = self.get_taskSet(self.randMethod)
            taskTensor = torch.tensor(taskLst, device=self.device, dtype=torch.float32)

            if self.task_cfg["task_w_randAdaptive"]:
                task_ratio = self.task_ratio/ torch.bincount(self.idx_envTask)

            self.idx_envTask = self.sample_task(ratio=1 / task_ratio**2)

            # ensure that all task_w are in idx
            for i in range(len(taskLst)):
                if i not in self.idx_envTask:
                    self.idx_envTask[i] = i

            wTrain = taskTensor[self.idx_envTask]

            if self.verbose:
                print("[Task] task ratio: ", task_ratio)
                print("[Task] counts: ", torch.bincount(self.idx_envTask))

            # # reset
            # self.task_ratio = torch.ones(len(taskLst), device=self.device)

        else:  # uniform random
            wTrain = torch.rand((self.n_env, self.dim), device=self.device)

        self.wTrain = self.normalize_task(wTrain)

class PointMassTask(SmartTask):
    def __init__(self, env_cfg) -> None:
        super().__init__(env_cfg)

    def define_task(self, c, dim, w):
        w_pos_norm = c[0] * [w[0]]
        w_vel = c[1] * dim * [w[1]]
        w_vel_norm = c[2] * [w[2]]
        w_prox = c[3] * [w[3]]
        return w_pos_norm + w_vel + w_vel_norm + w_prox


class PointerTask(SmartTask):
    def __init__(self, env_cfg) -> None:
        super().__init__(env_cfg)

    def define_task(self, c, d, w):
        w_px = c[0] * [w[0]]
        w_py = c[1] * [w[1]]
        w_vel_norm = c[2] * [w[2]]
        w_ang_norm = c[3] * [w[3]]
        w_angvel_norm = c[4] * [w[4]]
        return w_px + w_py + w_vel_norm + w_ang_norm + w_angvel_norm


def task_constructor(env_cfg, device):
    if "pointer" in env_cfg["env_name"].lower():
        return PointerTask(env_cfg, device)
    elif "pointmass" in env_cfg["env_name"].lower():
        return PointMassTask(env_cfg, device)
    else:
        print(f'task not implemented: {env_cfg["env_name"]}')
        return None
