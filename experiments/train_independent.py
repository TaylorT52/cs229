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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "independent")


def main():
    params = copy.deepcopy(flow_params)

    def env_creator(env_config):
        return MultiAgentPlatoonEnv(env_config)

    register_env("platoon_multiagent", env_creator)

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    lane_change_enabled = params["env"].additional_params.get("lane_change_enabled", False)
    if lane_change_enabled:
        num_features = 18 
    else:
        num_features = 9 
    
    obs_space = Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)
    
    if lane_change_enabled:
        act_space = Box(
            low=np.array([-3.0, -1.0], dtype=np.float32),
            high=np.array([3.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
    else:
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
        "use_ctde_obs": False,
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
                "count_steps_by": "agent_steps",
            },
            "lr": 3e-4,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.1, 
            "num_sgd_iter": 10,
            "sgd_minibatch_size": 128,
            "train_batch_size": 4000,
            "use_gae": True,
            "num_workers": 0,
            "framework": "torch",
            "log_level": "WARN",
        }
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("Starting Independent PPO Training")
    print("=" * 60)
    print(f"Training iterations: 150")
    print(f"Number of agents: {num_agents}")
    print(f"Features per agent: {num_features} ({'with' if lane_change_enabled else 'without'} lane change info)")
    print(f"Lane changes enabled: {lane_change_enabled}")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 60 + "\n")

    tune.run(
        ppo.PPOTrainer,
        stop={"training_iteration": 150},
        config=config,
        local_dir=RESULTS_DIR,
        loggers=DEFAULT_LOGGERS + (TBXLogger,),
        checkpoint_freq=10,
        checkpoint_at_end=True,
        verbose=1,
    )

    print("\nIndependent training complete!")
    ray.shutdown()

if __name__ == "__main__":
    main()
