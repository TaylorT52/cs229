"""Train script for CTDE model using RLlib PPO.

This script registers the custom CTDEModel and a multi-agent environment
that provides per-agent 'local' observations and a shared 'global' observation
for the centralized critic.

Key CTDE features:
- All agents share the SAME policy (centralized training)
- Each agent uses local observations (decentralized execution)
- Value function uses global/centralized state (centralized critic)
"""

import copy
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

experiments_dir = os.path.dirname(os.path.abspath(__file__))
if experiments_dir not in sys.path:
    sys.path.insert(0, experiments_dir)

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.tune.logger import DEFAULT_LOGGERS, TBXLogger
from ray.tune.registry import register_env

from ray.rllib.models import ModelCatalog

from multi_agent_env import MultiAgentPlatoonEnv
from platoon_config import flow_params
from ctde_policy import CTDEModel

from gym.spaces import Box, Dict as GymDict
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "ctde")


def main():
    # Register custom CTDE model
    ModelCatalog.register_custom_model("ctde_torch_model", CTDEModel)

    # Register multi-agent environment
    def env_creator(env_config):
        return MultiAgentPlatoonEnv(env_config)

    register_env("platoon_ctde", env_creator)

    # Initialize Ray
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    # Get number of agents from config (number of RL vehicles)
    num_agents = flow_params["veh"].num_rl_vehicles
    num_features = 5  # Features per agent (speed, headway, relative_speed, lane, normalized_speed)

    print(f"CTDE Configuration:")
    print(f"  - Number of agents (RL vehicles): {num_agents}")
    print(f"  - Features per agent: {num_features}")
    print(f"  - Using SHARED policy for all agents (CTDE)")

    # Build per-agent observation space (dict with local and global)
    obs_space = GymDict({
        "local": Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        "global": Box(low=-np.inf, high=np.inf, shape=(num_agents * num_features,), dtype=np.float32),
    })
    act_space = Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)

    # CTDE: ALL agents share the SAME policy
    # This is the key difference from independent learning!
    shared_policy_config = {
        "model": {
            "custom_model": "ctde_torch_model",
            "custom_model_config": {
                "global_obs_dim": num_agents * num_features,
                "local_obs_dim": num_features,
            },
            "fcnet_hiddens": [256, 256],  # Larger network for better learning
            "fcnet_activation": "tanh",
        }
    }

    # Single shared policy for all agents
    policies = {
        "shared_policy": (PPOTorchPolicy, obs_space, act_space, shared_policy_config)
    }

    # All agents map to the same shared policy
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        """Map all agents to shared policy (CTDE)."""
        return "shared_policy"

    # Create network instance for environment config
    from flow.networks import HighwayNetwork
    network = HighwayNetwork(
        name="highway",
        vehicles=flow_params["veh"],
        net_params=flow_params["net"],
        initial_config=flow_params["initial"],
    )

    config = ppo.DEFAULT_CONFIG.copy()
    config.update({
        "env": "platoon_ctde",
        "env_config": {
            "env_params": flow_params["env"],
            "sim_params": flow_params["sim"],
            "network": network,  # Pass network instance
            "simulator": flow_params.get("simulator", "traci"),
        },
        "framework": "torch",
        "num_workers": 0,
        "horizon": flow_params["env"].horizon,
        
        # CTDE multi-agent configuration
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["shared_policy"],  # Only train the shared policy
            "count_steps_by": "agent_steps",
        },
        
        # PPO hyperparameters
        "lr": 3e-4,
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "num_sgd_iter": 10,
        "sgd_minibatch_size": 128,
        "train_batch_size": 4000,
        "use_gae": True,
        
        "log_level": "WARN",
    })

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("Starting CTDE PPO Training")
    print("=" * 60)
    print(f"Training iterations: 25")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 60 + "\n")

    tune.run(
        ppo.PPOTrainer,
        stop={"training_iteration": 25},
        config=config,
        local_dir=RESULTS_DIR,
        loggers=DEFAULT_LOGGERS + (TBXLogger,),
        checkpoint_freq=20,
        checkpoint_at_end=True,
        verbose=1,
    )

    print("\nCTDE training complete!")
    ray.shutdown()


if __name__ == "__main__":
    main()
