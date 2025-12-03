"""Visualize trained CTDE policy in SUMO GUI.

This script loads a trained CTDE checkpoint and runs it with visualization
so you can see how the trained agents behave.

Usage:
    python visualize_trained_ctde.py --checkpoint <path_to_checkpoint>
    
Example:
    python visualize_trained_ctde.py --checkpoint ../results/ctde/PPO_2025-12-02_18-13-08/PPO_platoon_ctde_*/checkpoint_200
"""

import sys
import os
import argparse
import glob

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

experiments_dir = os.path.dirname(os.path.abspath(__file__))
if experiments_dir not in sys.path:
    sys.path.insert(0, experiments_dir)

import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from multi_agent_env import MultiAgentPlatoonEnv
from platoon_config import flow_params
from ctde_policy import CTDEModel
from gym.spaces import Box, Dict as GymDict
import numpy as np


def find_latest_checkpoint(results_dir=None):
    """Automatically find the latest checkpoint from results directory."""
    if results_dir is None:
        # Default to results/ctde relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(project_root, "results", "ctde")
    
    results_dir = os.path.expanduser(results_dir)
    
    # Find all checkpoint files
    pattern = os.path.join(results_dir, "PPO_*/PPO_*/checkpoint_*/checkpoint-*")
    checkpoint_files = glob.glob(pattern)
    
    # Filter out metadata files
    checkpoint_files = [f for f in checkpoint_files if not f.endswith(".tune_metadata")]
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time, get most recent
    latest = max(checkpoint_files, key=os.path.getmtime)
    return latest


def find_checkpoint(checkpoint_path):
    """Find the actual checkpoint file."""
    # Expand user and resolve path
    checkpoint_path = os.path.expanduser(checkpoint_path)
    
    # If it's a directory, look for checkpoint file inside
    if os.path.isdir(checkpoint_path):
        # Look for checkpoint-XX file directly in directory
        checkpoint_files = [f for f in os.listdir(checkpoint_path) 
                          if f.startswith("checkpoint-") and not f.endswith(".tune_metadata")]
        if checkpoint_files:
            full_path = os.path.join(checkpoint_path, checkpoint_files[0])
            if os.path.exists(full_path):
                return full_path
        
        # Or look for checkpoint subdirectory (checkpoint_000200 format)
        subdirs = [d for d in os.listdir(checkpoint_path) if d.startswith("checkpoint_")]
        if subdirs:
            # Sort by iteration number
            def get_iteration(d):
                try:
                    return int(d.split("_")[-1])
                except:
                    return 0
            latest_subdir = max(subdirs, key=get_iteration)
            checkpoint_num = latest_subdir.split("_")[-1]
            checkpoint_file = os.path.join(checkpoint_path, latest_subdir, f"checkpoint-{checkpoint_num}")
            if os.path.exists(checkpoint_file):
                return checkpoint_file
    
    # If it's already a file, return as is
    if os.path.isfile(checkpoint_path):
        return checkpoint_path
    
    # Try glob pattern
    matches = glob.glob(checkpoint_path)
    if matches:
        return matches[0]
    
    return checkpoint_path


def visualize_policy(checkpoint_path, num_episodes=3, horizon=None):
    """Load trained policy and visualize it in SUMO.
    
    Args:
        checkpoint_path: Path to checkpoint
        num_episodes: Number of episodes to run
        horizon: Episode length in steps (None = use default from config)
    """
    
    # Register model
    ModelCatalog.register_custom_model("ctde_torch_model", CTDEModel)
    
    # Register environment
    def env_creator(env_config):
        return MultiAgentPlatoonEnv(env_config)
    
    register_env("platoon_ctde", env_creator)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    
    # Get configuration
    num_agents = flow_params["veh"].num_rl_vehicles
    num_features = 5
    
    obs_space = GymDict({
        "local": Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        "global": Box(low=-np.inf, high=np.inf, shape=(num_agents * num_features,), dtype=np.float32),
    })
    act_space = Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
    
    # Create network
    from flow.networks import HighwayNetwork
    network = HighwayNetwork(
        name="highway",
        vehicles=flow_params["veh"],
        net_params=flow_params["net"],
        initial_config=flow_params["initial"],
    )
    
    # CTDE policy configuration
    from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
    
    shared_policy_config = {
        "model": {
            "custom_model": "ctde_torch_model",
            "custom_model_config": {
                "global_obs_dim": num_agents * num_features,
                "local_obs_dim": num_features,
            },
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "tanh",
        }
    }
    
    policies = {
        "shared_policy": (PPOTorchPolicy, obs_space, act_space, shared_policy_config)
    }
    
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"
    
    # Enable rendering - create new SumoParams with render=True
    from flow.core.params import SumoParams, EnvParams
    sim_params = SumoParams(
        render=True,  # Enable visualization
        sim_step=flow_params["sim"].sim_step,
        restart_instance=flow_params["sim"].restart_instance,
    )
    
    # Override horizon if specified
    if horizon is not None:
        # Create new EnvParams with custom horizon
        # Copy any additional_params if they exist
        env_params_dict = {"horizon": horizon}
        if hasattr(flow_params["env"], "additional_params"):
            env_params_dict["additional_params"] = flow_params["env"].additional_params
        env_params = EnvParams(**env_params_dict)
    else:
        env_params = flow_params["env"]
    
    config = {
        "env": "platoon_ctde",
        "env_config": {
            "env_params": env_params,  # Use custom horizon if provided
            "sim_params": sim_params,  # With rendering enabled
            "network": network,
            "simulator": "traci",
        },
        "framework": "torch",
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
    }
    
    # Find checkpoint file
    checkpoint_file = find_checkpoint(checkpoint_path)
    print(f"Loading checkpoint from: {checkpoint_file}")
    
    if not os.path.exists(checkpoint_file):
        print(f"Error: Checkpoint not found: {checkpoint_file}")
        print(f"Tried path: {checkpoint_path}")
        ray.shutdown()
        return
    
    # Create trainer and restore
    trainer = ppo.PPOTrainer(config=config)
    trainer.restore(checkpoint_file)
    
    # Calculate simulation duration
    sim_step = sim_params.sim_step
    episode_horizon = env_params.horizon
    episode_duration_sec = episode_horizon * sim_step
    total_duration_sec = episode_duration_sec * num_episodes
    
    print("=" * 60)
    print("Visualizing Trained CTDE Policy")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_file}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Steps per episode: {episode_horizon}")
    print(f"Simulation step: {sim_step}s")
    print(f"Duration per episode: {episode_duration_sec:.1f}s ({episode_duration_sec/60:.1f} minutes)")
    print(f"Total duration: {total_duration_sec:.1f}s ({total_duration_sec/60:.1f} minutes)")
    print(f"RL vehicles (red): {num_agents}")
    print(f"Human vehicles (blue): {flow_params['veh'].num_vehicles - num_agents}")
    print("=" * 60)
    print("\nStarting visualization...")
    print("The SUMO GUI should open shortly.")
    print("Watch the RED vehicles - they are using the trained CTDE policy!")
    print("Close the SUMO GUI window to stop.\n")
    
    # Run episodes
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        # Create environment
        env = MultiAgentPlatoonEnv(config["env_config"])
        obs = env.reset()
        
        done = {"__all__": False}
        step = 0
        total_reward = {agent_id: 0.0 for agent_id in env.agent_ids}
        
        while not done["__all__"]:
            # Get actions from trained policy
            actions = {}
            for agent_id, agent_obs in obs.items():
                policy_id = policy_mapping_fn(agent_id, None, None)
                action = trainer.compute_action(
                    agent_obs, 
                    policy_id=policy_id
                )
                actions[agent_id] = action
            
            # Step environment
            obs, rewards, done, info = env.step(actions)
            
            # Accumulate rewards
            for agent_id, reward in rewards.items():
                total_reward[agent_id] += reward
            
            step += 1
            
            if step % 200 == 0:
                avg_reward = np.mean(list(total_reward.values()))
                print(f"  Step {step}, Avg Reward: {avg_reward:.2f}")
        
        avg_reward = np.mean(list(total_reward.values()))
        print(f"  Episode complete after {step} steps")
        print(f"  Average reward per agent: {avg_reward:.2f}")
        
        env.close()
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    
    ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Visualize trained CTDE policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (auto-detects latest if not provided)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run (default: 3)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory to search for checkpoints (default: ../results/ctde)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Episode duration in minutes (e.g., --duration 5 for 5 minutes). Overrides default horizon."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Episode length in simulation steps (overrides --duration if both specified)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect checkpoint if not provided
    if args.checkpoint is None:
        print("No checkpoint specified, searching for latest checkpoint...")
        checkpoint = find_latest_checkpoint(args.results_dir)
        if checkpoint is None:
            print("Error: No checkpoint found!")
            print("Please specify --checkpoint or ensure results exist in results/ctde/")
            return
        print(f"Found latest checkpoint: {checkpoint}")
    else:
        checkpoint = args.checkpoint
    
    # Calculate horizon from duration if specified
    horizon = args.horizon
    if horizon is None and args.duration is not None:
        # Default sim_step is 0.1 seconds
        sim_step = flow_params["sim"].sim_step
        horizon = int(args.duration * 60 / sim_step)  # Convert minutes to steps
        print(f"Setting horizon to {horizon} steps for {args.duration} minute episode")
    
    visualize_policy(checkpoint, args.episodes, horizon=horizon)


if __name__ == "__main__":
    main()

