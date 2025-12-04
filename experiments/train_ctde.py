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
    
    # Check if lane changes are enabled
    lane_change_enabled = flow_params["env"].additional_params.get("lane_change_enabled", False)

    # Features per agent: must match PlatoonEnv.num_features
    # We use 9 rich longitudinal features when lane changes are disabled:
    #   speed, headway, relative_speed, lane, normalized_speed,
    #   accel, ttc, leader_speed_norm, is_slow_leader
    # and 18 when lane changes are enabled (adds 9 lateral features).
    if lane_change_enabled:
        num_features = 18
    else:
        num_features = 9

    print(f"CTDE Configuration:")
    print(f"  - Number of agents (RL vehicles): {num_agents}")
    print(f"  - Features per agent: {num_features} ({'with' if lane_change_enabled else 'without'} lane change info)")
    print(f"  - Using SHARED policy for all agents (CTDE)")
    print(f"  - Lane changes enabled: {lane_change_enabled}")

    # Build per-agent observation space (dict with local and global)
    obs_space = GymDict({
        "local": Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        "global": Box(low=-np.inf, high=np.inf, shape=(num_agents * num_features,), dtype=np.float32),
    })
    
    # Action space depends on whether lane changes are enabled
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

    # CTDE: ALL agents share the SAME policy
    # This is the key difference from independent learning!
    shared_policy_config = {
        "model": {
            "custom_model": "ctde_torch_model",
            "custom_model_config": {
                "global_obs_dim": num_agents * num_features,
                "local_obs_dim": num_features,
                "action_space_size": 2 if lane_change_enabled else 1,
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
            "use_ctde_obs": True,  # Enable CTDE observation format (Dict with local/global)
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
        "clip_param": 0.1,  # Reduced from 0.2 for more stable learning with high-variance lane changes
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
    print(f"Training iterations: 150")
    print(f"Lane changes enabled: {lane_change_enabled}")
    print(f"Action space: {act_space}")
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

    print("\nCTDE training complete!")
    ray.shutdown()


if __name__ == "__main__":
    main()
