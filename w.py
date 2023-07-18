import collections.abc


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def fix_wandb(d):
    ret = {}
    d = dict(d)
    for k, v in d.items():
        if not "." in k:
            ret[k] = v
        else:
            ks = k.split(".")
            a = fix_wandb({".".join(ks[1:]): v})
            if ks[0] not in ret:
                ret[ks[0]] = {}
            update_dict(ret[ks[0]], a)
    return ret


x = {
    "agent.lr": 0.004548949491782365,
    "agent.policy_lr": 0.002699699083683589,
    "buffer.mini_batch_size": 4096,
    "env.num_envs": 1024,
    "env.task.rand_vel_targets": False,
    "env.task.rand_weights": True,
}

a = fix_wandb(x)
print(a)
