"""Train independent PPO policies for multi-agent platooning.

Each RL vehicle gets its own policy and reward signal, enabling
comparison of independent learning against the joint controller.
"""

import copy
import os
import numpy as np

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.logger import DEFAULT_LOGGERS, TBXLogger
from ray.tune.registry import register_env
from gym.spaces import Box

from flow.networks import HighwayNetwork
from multi_agent_env import MultiAgentPlatoonEnv
from platoon_config import flow_params

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "multi_agent")


def main():
    params = copy.deepcopy(flow_params)

    def env_creator(env_config):
        return MultiAgentPlatoonEnv(env_config)

    register_env("platoon_multiagent", env_creator)

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    # Check if lane changes are enabled to determine observation space
    lane_change_enabled = params["env"].additional_params.get("lane_change_enabled", False)
    if lane_change_enabled:
        num_features = 18  # 9 longitudinal + 9 lateral
    else:
        num_features = 9  # 9 longitudinal features only
    
    obs_space = Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)
    
    # Check if lane changes are enabled
    if lane_change_enabled:
        # Actions: [accel, lane_change] per agent
        act_space = Box(
            low=np.array([-3.0, -1.0], dtype=np.float32),
            high=np.array([3.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
    else:
        # Just acceleration
        act_space = Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)

    num_agents = params["veh"].num_rl_vehicles
    policies = {
        f"agent_{i}": (None, obs_space, act_space, {})
        for i in range(num_agents)
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return agent_id

    network = HighwayNetwork(
        name="highway",
        vehicles=params["veh"],
        net_params=params["net"],
        initial_config=params["initial"],
    )

    env_config = {
        "env_params": params["env"],
        "sim_params": params["sim"],
        "network": network,
        "simulator": "traci",
    }

    config = ppo.DEFAULT_CONFIG.copy()
    config.update(
        {
            "env": "platoon_multiagent",
            "env_config": env_config,
            "horizon": params["env"].horizon,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": list(policies.keys()),
                "count_steps_by": "env_steps",
            },
            "train_batch_size": 4000,
            "sgd_minibatch_size": 128,
            "num_sgd_iter": 10,
            "clip_param": 0.2,
            "num_workers": 0,
            "framework": "torch",
            "log_level": "WARN",
            "lr": 5e-5,
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
