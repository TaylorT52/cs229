"""Train a single joint policy (independent control) for the platoon env."""

import copy
import os

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.logger import DEFAULT_LOGGERS, TBXLogger
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env

from .platoon_config import flow_params

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "independent")


def main():
    params = copy.deepcopy(flow_params)
    create_env, env_name = make_create_env(params, version=0)
    register_env(env_name, create_env)

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    config = ppo.DEFAULT_CONFIG.copy()
    config.update(
        {
            "env": env_name,
            "num_workers": 0,
            "framework": "torch",
            "horizon": params["env"].horizon,
            "train_batch_size": 4000,
            "sgd_minibatch_size": 128,
            "num_sgd_iter": 10,
            "clip_param": 0.2,
            "log_level": "WARN",
            "env_config": {"flow_params": params, "run": 0},
        }
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    tune.run(
        ppo.PPOTrainer,
        stop={"training_iteration": 100},
        config=config,
        local_dir=RESULTS_DIR,
        loggers=DEFAULT_LOGGERS + (TBXLogger,),
        checkpoint_freq=10,
        checkpoint_at_end=True,
    )

    ray.shutdown()


if __name__ == "__main__":
    main()
