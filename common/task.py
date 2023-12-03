import torch
import itertools


class TaskObject:
    def __init__(self, initW, n_env, randTasks, taskSetType, task_cfg, device) -> None:
        self.n_env = n_env
        self.randTasks = randTasks
        self.taskSetType = taskSetType
        self.task_cfg = task_cfg
        self.device = device

        self.initW = torch.tensor(initW, device=self.device, dtype=torch.float32)
        self.dim = int(self.initW.shape[0])

        self.W = self.normalize_task(torch.tile(self.initW, (self.n_env, 1)))  # [N, F]

        if self.randTasks:
            if self.taskSetType != "uniform":
                self.taskSet = self.define_taskSet(self.taskSetType)
                self.reset_taskRatio()
            self.W = self.sample_tasks()

    def define_taskSet(self, taskSetType):
        taskset = self.task_cfg["taskSet"]
        if taskSetType == "permute":
            taskSet = list(itertools.product([0, 1], repeat=self.dim))
            taskSet.pop(0)  # remove all zero vector
        elif taskSetType == "identity":
            taskSet = [
                [1 if i == j else 0 for j in range(self.dim)] for i in range(self.dim)
            ]
        elif taskSetType == "achievable":
            taskSet = taskset["achievable"]
        elif taskSetType == "redundant":
            taskSet = taskset["redundant"]
        elif taskSetType == "single":
            taskSet = taskset["single"]
        else:
            raise ValueError(f"Warning: {taskSetType} is not implemented")
        return torch.tensor(taskSet, dtype=torch.float32, device=self.device)

    def sample_tasks(self):
        if self.taskSetType == "uniform":
            tasks = torch.rand((self.n_env, self.dim), device=self.device)
        else:
            id = self.sample_taskID(self.taskRatio)

            assert len(self.taskSet) <= len(id), f"num envs {len(id)} is less than num taskSet {len(self.taskSet)}"

            # ensure that all task are in id
            for i in range(len(self.taskSet)):
                if i not in id:
                    id[i] = i

            self.update_id(id)
            tasks = self.taskSet[id]

        return self.normalize_task(tasks)

    def update_id(self, id):
        self.id = id

    def normalize_task(self, w):
        w /= w.norm(1, 1, keepdim=True)
        return w

    def sample_taskID(self, ratio):
        """sample tasks based on their ratio"""
        return torch.multinomial(ratio, self.n_env, replacement=True)

    def reset_taskRatio(self):
        self.taskRatio = torch.ones(len(self.taskSet), device=self.device) / len(
            self.taskSet
        )

    def add_task(self, w: torch.tensor):
        w = w.view(-1, self.dim)
        self.taskSet = torch.cat([self.taskSet, w], 0)
        self.reset_taskRatio()
        self.W = self.sample_tasks()


class SmartTask:
    def __init__(self, env_cfg, device) -> None:
        self.env_cfg = env_cfg
        self.task_cfg = self.env_cfg["task"]
        self.device = device
        self.verbose = self.task_cfg.get("verbose", False)

        self.n_env = self.env_cfg["num_envs"]
        use_feature = self.env_cfg["feature"]["use_feature"]
        env_dim = self.env_cfg["feature"]["dim"]
        self.randTasks = self.task_cfg.get("rand_task", False)
        self.taskSet_train = self.task_cfg.get("taskSet_train", None)
        self.taskSet_eval = self.task_cfg.get("taskSet_eval", None)
        self.intervalWeightRand = self.task_cfg.get("intervalWeightRand", 2)

        wTrain = self.define_task(use_feature, env_dim, self.task_cfg["task_wTrain"])
        wEval = self.define_task(use_feature, env_dim, self.task_cfg["task_wEval"])

        self.Train = TaskObject(
            wTrain,
            self.n_env,
            self.randTasks,
            self.taskSet_train,
            self.task_cfg,
            device,
        )
        self.Eval = TaskObject(
            wEval, self.n_env, self.randTasks, self.taskSet_eval, self.task_cfg, device
        )

        self.dim = int(self.Train.dim)

        if self.verbose:
            print("[Task] training task set: \n", self.Train.taskSetType)
            print("[Task] training tasks: \n", self.Train.W)
            print("[Task] evaluation task set: \n", self.Eval.taskSetType)
            print("[Task] evaluation tasks: \n", self.Eval.W)
            print("\n")

    def define_task(self):
        """define initial task weight as a vector"""
        NotImplementedError

    def rand_task(self, episodes):
        if (
            ((episodes - 1) % self.intervalWeightRand == 0)
            and (self.env_cfg["mode"] == "train")
            and (self.randTasks)
        ):
            self.Train.sample_tasks()

            if self.verbose:
                print("[Task] sample new tasks:")
                print("[Task] Train.W[0]: ", self.Train.W[0])
                print("[Task] Train.taskRatio: ", self.Train.taskRatio)
                print("[Task] Train Task Counts: ", torch.bincount(self.Train.id))
                print("[Task] Eval.W[0]: ", self.Eval.W[0])
                print("[Task] Train.taskRatio: ", self.Eval.taskRatio)
                print("[Task] Eval Task Counts: ", torch.bincount(self.Eval.id))
                print("\n")

    def adapt_task(self, episode_r):
        """
        Update task ratio based on reward.
        The more reward the less likely for a task to be sampled.
        """
        task_performance = self.Train.taskRatio.index_add(
            dim=0, index=self.Train.id, source=episode_r.float()
        ) / torch.bincount(self.Train.id)

        new_ratio = task_performance**-1
        new_ratio /= new_ratio.norm(1, keepdim=True)
        self.Train.taskRatio = new_ratio

        if self.verbose:
            print(
                f"[Task] updated task ratio: {new_ratio} \n as inverse of return {task_performance} \n"
            )

    def add_task(self, w: torch.tensor):
        self.Train.add_task(w)
        if self.verbose:
            print(f"[Task] new task {w} added to train task set \n")


class PointMassTask(SmartTask):
    def __init__(self, env_cfg, device) -> None:
        super().__init__(env_cfg, device)

    def define_task(self, c, dim, w):
        w_pos_norm = c[0] * [w[0]]
        w_vel = c[1] * dim * [w[1]]
        w_vel_norm = c[2] * [w[2]]
        w_prox = c[3] * [w[3]]
        return w_pos_norm + w_vel + w_vel_norm + w_prox


class PointerTask(SmartTask):
    def __init__(self, env_cfg, device) -> None:
        super().__init__(env_cfg, device)

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
