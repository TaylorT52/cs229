"""Train 10 independent PPO policies for multi-agent platooning.

Each of the 10 RL vehicles has its own policy that learns independently
using only its local observations and individual rewards.
"""

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
    # Register multi-agent environment
    def env_creator(env_config):
        return MultiAgentPlatoonEnv(env_config)
    
    register_env("platoon_multiagent", env_creator)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    
    # Define observation and action spaces (same for all agents)
    obs_space = Box(
        low=-np.inf, 
        high=np.inf, 
        shape=(5,),  # 5 features per agent
        dtype=np.float32
    )
    act_space = Box(
        low=-3.0, 
        high=3.0, 
        shape=(1,),  # 1 action per agent (acceleration)
        dtype=np.float32
    )
    
    # Create independent policies (one per RL vehicle)
    # Must match num_vehicles in platoon_config.py (currently 2)
    num_agents = 2  # Reduced from 10 for faster training
    policies = {
        f"agent_{i}": (
            None,           # Use default PPO policy class
            obs_space,      # Observation space
            act_space,      # Action space
            {}              # Config overrides (empty = use defaults)
        )
        for i in range(num_agents)
    }
    
    # Policy mapping function: each agent uses its own policy
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        """Map agent_id to policy_id.
        
        For independent learning, each agent uses its own policy.
        agent_0 → policy "agent_0"
        agent_1 → policy "agent_1"
        etc.
        """
        return agent_id
    
    
    # Create Flow network
    network = HighwayNetwork(
        name="highway",
        vehicles=flow_params["veh"],
        net_params=flow_params["net"],
        initial_config=flow_params["initial"],
    )
    
    env_config = {
        "env_params": flow_params["env"],
        "sim_params": flow_params["sim"],
        "network": network,
        "simulator": "traci",
    }
    
    
    config = ppo.DEFAULT_CONFIG.copy()
    config.update({
        # Environment
        "env": "platoon_multiagent",
        "env_config": env_config,
        "horizon": 500,  # Reduced from 1500 for faster training (50 seconds sim)
        
        # Multi-agent settings
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": [f"agent_{i}" for i in range(num_agents)],
            "count_steps_by": "agent_steps",
        },
        
        # PPO hyperparameters
        "lr": 5e-4,                    # Increased learning rate for faster learning
        "gamma": 0.99,                 # Discount factor
        "lambda": 0.95,                # GAE parameter
        "clip_param": 0.2,             # PPO clipping parameter
        "num_sgd_iter": 5,             # Reduced from 10 for faster iterations
        "sgd_minibatch_size": 128,     # Mini-batch size
        "train_batch_size": 2000,      # Reduced from 4000 for faster iterations
        
        # Neural network architecture
        "model": {
            "fcnet_hiddens": [32, 32],  # Smaller network for faster training
            "fcnet_activation": "relu",
        },
        
        # Training settings
        "num_workers": 0,              # Number of parallel workers (0 = local)
        "framework": "torch",          # Use PyTorch
        "log_level": "WARN",
    })
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Starting Independent PPO Training")
    print("=" * 60)
    print(f"Number of agents: {num_agents}")
    print(f"Number of policies: {len(policies)}")
    print(f"Observation space per agent: {obs_space.shape}")
    print(f"Action space per agent: {act_space.shape}")
    print(f"Training iterations: 30 (fast mode)")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 60)
    
    tune.run(
        ppo.PPOTrainer,
        stop={"training_iteration": 30},  # Reduced from 100 for faster testing
        config=config,
        local_dir=RESULTS_DIR,
        loggers=DEFAULT_LOGGERS + (TBXLogger,),
        checkpoint_freq=5,            # Save checkpoint every 5 iterations
        checkpoint_at_end=True,
        verbose=1,
    )
    
    print("\nTraining complete!")
    print(f"Checkpoints saved in: {RESULTS_DIR}")
    
    ray.shutdown()


if __name__ == "__main__":
    main()
