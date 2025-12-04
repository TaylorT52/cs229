"""Evaluate a trained PPO model using config from params.json.

Automatically loads the training config from params.json to ensure
the model architecture matches the checkpoint.
"""

import os
import glob
import json
import numpy as np

import ray
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from gym.spaces import Box

from flow.networks import HighwayNetwork
from flow.core.params import SumoParams
from multi_agent_env import MultiAgentPlatoonEnv
from platoon_config import flow_params

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "multi_agent")


def find_latest_checkpoint_and_params():
    """Find the most recent training run, checkpoint, and params.json."""
    # Get the most recent training directory
    training_dirs = glob.glob(os.path.join(RESULTS_DIR, "PPO_*"))
    if not training_dirs:
        raise FileNotFoundError("No training results found in results/multi_agent/")
    
    latest_dir = max(training_dirs, key=os.path.getctime)
    print(f"Found training directory: {latest_dir}")
    
    # Find the trial subdirectory
    trial_dirs = [d for d in os.listdir(latest_dir) 
                  if os.path.isdir(os.path.join(latest_dir, d)) and d.startswith("PPO_")]
    
    if not trial_dirs:
        raise FileNotFoundError(f"No trial directory found in {latest_dir}")
    
    trial_dir = os.path.join(latest_dir, trial_dirs[0])
    print(f"Found trial directory: {trial_dir}")
    
    # Load params.json
    params_path = os.path.join(trial_dir, "params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.json not found in {trial_dir}")
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    print(f"Loaded params.json")
    
    # Find the latest checkpoint
    checkpoint_dirs = glob.glob(os.path.join(trial_dir, "checkpoint_*"))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {trial_dir}")
    
    latest_checkpoint_dir = max(checkpoint_dirs, 
                                 key=lambda x: int(x.split("_")[-1]))
    
    checkpoint_num = int(latest_checkpoint_dir.split("_")[-1])
    checkpoint_path = os.path.join(latest_checkpoint_dir, f"checkpoint-{checkpoint_num}")
    
    print(f"Found checkpoint: {checkpoint_path}")
    return checkpoint_path, params


def create_env_config(render=True):
    """Create environment config with or without rendering."""
    network = HighwayNetwork(
        name="highway",
        vehicles=flow_params["veh"],
        net_params=flow_params["net"],
        initial_config=flow_params["initial"],
    )
    
    sim_params = SumoParams(
        render=render,
        sim_step=0.1,
        restart_instance=True,
    )
    
    return {
        "env_params": flow_params["env"],
        "sim_params": sim_params,
        "network": network,
        "simulator": "traci",
    }


def main():
    """Load trained model and run visualization."""
    
    print("=" * 60)
    print("Evaluating Trained PPO Model")
    print("=" * 60)
    
    # Find checkpoint and load params.json
    checkpoint_path, saved_params = find_latest_checkpoint_and_params()
    
    # Extract key settings from params.json
    model_config = saved_params.get("model", {})
    multiagent_config = saved_params.get("multiagent", {})
    horizon = saved_params.get("horizon", 1500)
    framework = saved_params.get("framework", "torch")
    
    print(f"\nLoaded config from params.json:")
    print(f"  - Model: fcnet_hiddens={model_config.get('fcnet_hiddens')}")
    print(f"  - Activation: {model_config.get('fcnet_activation')}")
    print(f"  - Horizon: {horizon}")
    print(f"  - Framework: {framework}")
    
    # Get number of agents from saved policies
    saved_policies = multiagent_config.get("policies", {})
    num_agents = len(saved_policies)
    print(f"  - Number of agents: {num_agents}")
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    
    # Register environment
    def env_creator(env_config):
        return MultiAgentPlatoonEnv(env_config)
    register_env("platoon_multiagent", env_creator)
    
    # Define spaces
    obs_space = Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
    act_space = Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
    
    # Create policies matching the saved config
    policies = {
        f"agent_{i}": (None, obs_space, act_space, {})
        for i in range(num_agents)
    }
    
    def policy_mapping_fn(agent_id, **kwargs):
        return agent_id
    
    # Build config using params.json values
    config = ppo.DEFAULT_CONFIG.copy()
    config.update({
        "env": "platoon_multiagent",
        "env_config": create_env_config(render=True),
        "horizon": horizon,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "model": model_config,  # Use model config from params.json!
        "num_workers": 0,
        "framework": framework,
    })
    
    # Create trainer and restore
    print("\nLoading trained model...")
    trainer = ppo.PPOTrainer(config=config)
    trainer.restore(checkpoint_path)
    print("Model loaded successfully!")
    
    # Create evaluation environment (with GUI)
    print("\nStarting simulation...")
    env = MultiAgentPlatoonEnv(create_env_config(render=True))
    
    # Run evaluation episodes
    num_episodes = 3
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        obs = env.reset()
        done = {"__all__": False}
        total_reward = {f"agent_{i}": 0.0 for i in range(num_agents)}
        step = 0
        
        while not done["__all__"]:
            # Get actions from trained policies
            actions = {}
            for agent_id in obs.keys():
                action = trainer.compute_single_action(
                    obs[agent_id],
                    policy_id=agent_id,
                )
                actions[agent_id] = action
            
            # Step environment
            obs, rewards, done, info = env.step(actions)
            
            # Accumulate rewards
            for agent_id, reward in rewards.items():
                total_reward[agent_id] += reward
            
            step += 1
            
            if step % 100 == 0:
                avg_reward = np.mean(list(total_reward.values()))
                print(f"  Step {step}: avg reward = {avg_reward:.2f}")
        
        # Episode summary
        print(f"\nEpisode {episode + 1} complete!")
        print(f"  Total steps: {step}")
        for agent_id, reward in total_reward.items():
            print(f"  {agent_id}: {reward:.2f}")
        print(f"  Average: {np.mean(list(total_reward.values())):.2f}")
    
    # Cleanup
    env.env.terminate()
    ray.shutdown()
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
