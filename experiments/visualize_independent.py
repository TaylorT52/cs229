import os
import sys
import argparse
import glob
import json
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

experiments_dir = os.path.dirname(os.path.abspath(__file__))
if experiments_dir not in sys.path:
    sys.path.insert(0, experiments_dir)

import ray
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from gym.spaces import Box

from flow.networks import HighwayNetwork
from multi_agent_env import MultiAgentPlatoonEnv
from platoon_config import flow_params


def find_latest_checkpoint(results_dir=None):
    """Find the latest checkpoint from training runs.
    
    Args:
        results_dir: Directory to search for checkpoints. If None, uses results/multi_agent
    
    Returns:
        tuple: (checkpoint_dir, checkpoint_num) or (None, None) if not found
    """
    if results_dir is None:
        results_dir = os.path.join(project_root, "results", "multi_agent")
    
    if not os.path.exists(results_dir):
        return None, None
    
    # Find all experiment directories
    exp_dirs = glob.glob(os.path.join(results_dir, "PPO_*"))
    if not exp_dirs:
        return None, None
    
    # Get the most recent one
    latest_exp = max(exp_dirs, key=os.path.getmtime)
    
    # Find all trial directories
    trial_dirs = glob.glob(os.path.join(latest_exp, "PPO_*"))
    if not trial_dirs:
        return None, None
    
    latest_trial = max(trial_dirs, key=os.path.getmtime)
    
    # Find all checkpoints
    checkpoint_dirs = glob.glob(os.path.join(latest_trial, "checkpoint_*"))
    if not checkpoint_dirs:
        return None, None
    
    # Get the latest checkpoint
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
    checkpoint_num = int(os.path.basename(latest_checkpoint).split("_")[1])
    
    return latest_checkpoint, checkpoint_num


def load_trained_model(checkpoint_path, checkpoint_num=None):
    """Load a trained PPO model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory or trial directory
        checkpoint_num: Checkpoint number to load (e.g., 30 for checkpoint_000030)
                        If None, loads the latest checkpoint
    
    Returns:
        PPOTrainer: Loaded trainer with trained policies
    """
    # Check if checkpoint_path is already a checkpoint directory
    is_checkpoint_dir = os.path.basename(checkpoint_path).startswith("checkpoint_")
    
    if is_checkpoint_dir:
        # checkpoint_path is already a checkpoint directory
        checkpoint_dir = checkpoint_path
        checkpoint_num = int(os.path.basename(checkpoint_dir).split("_")[1])
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_num}")
    elif checkpoint_num is not None:
        # checkpoint_path is a trial directory, construct checkpoint path
        checkpoint_dir = os.path.join(checkpoint_path, f"checkpoint_{checkpoint_num:06d}")
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_num}")
    else:
        # Find latest checkpoint in trial directory
        checkpoint_dirs = glob.glob(os.path.join(checkpoint_path, "checkpoint_*"))
        if not checkpoint_dirs:
            raise ValueError(f"No checkpoints found in {checkpoint_path}")
        checkpoint_dir = max(checkpoint_dirs, key=os.path.getmtime)
        checkpoint_num = int(os.path.basename(checkpoint_dir).split("_")[1])
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_num}")
    
    if not os.path.exists(checkpoint_file):
        raise ValueError(f"Checkpoint file not found: {checkpoint_file}")
    
    print(f"Loading checkpoint: {checkpoint_file}")
    
    # Load saved config from params.json (in trial directory)
    # Find trial directory (parent of checkpoint directory)
    trial_dir = os.path.dirname(checkpoint_dir) if is_checkpoint_dir else checkpoint_path
    params_file = os.path.join(trial_dir, "params.json")
    
    saved_config = {}
    if os.path.exists(params_file):
        print(f"Loading config from: {params_file}")
        with open(params_file, 'r') as f:
            saved_config = json.load(f)
    else:
        print(f"Warning: params.json not found at {params_file}, using defaults")
    
    # Register environment
    def env_creator(env_config):
        return MultiAgentPlatoonEnv(env_config)
    
    register_env("platoon_multiagent", env_creator)
    
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
    
    # Define observation and action spaces
    num_agents = flow_params["veh"].num_rl_vehicles
    obs_space = Box(
        low=-np.inf, 
        high=np.inf, 
        shape=(5,),
        dtype=np.float32
    )
    act_space = Box(
        low=-3.0, 
        high=3.0, 
        shape=(1,),
        dtype=np.float32
    )
    
    # Create policies
    policies = {
        f"agent_{i}": (
            None,
            obs_space,
            act_space,
            {}
        )
        for i in range(num_agents)
    }
    
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return agent_id
    
    # Create config - use saved config if available, otherwise use defaults
    config = ppo.DEFAULT_CONFIG.copy()
    
    # Extract model config from saved config
    model_config = saved_config.get("model", {})
    
    # Update with saved config values (prioritize saved config)
    config.update({
        "env": "platoon_multiagent",
        "env_config": env_config,
        "horizon": saved_config.get("horizon", flow_params["env"].horizon),
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "num_workers": 0,
        "framework": saved_config.get("framework", "torch"),
        "log_level": "ERROR",
        "model": model_config,  # Use saved model architecture
    })
    
    # Copy other relevant config from saved config
    if "lr" in saved_config:
        config["lr"] = saved_config["lr"]
    if "gamma" in saved_config:
        config["gamma"] = saved_config["gamma"]
    if "lambda" in saved_config:
        config["lambda"] = saved_config["lambda"]
    if "clip_param" in saved_config:
        config["clip_param"] = saved_config["clip_param"]
    
    # Create trainer and restore checkpoint
    trainer = ppo.PPOTrainer(config=config)
    trainer.restore(checkpoint_file)
    
    print(f"Successfully loaded checkpoint {checkpoint_num}")
    return trainer, env_config


def run_visualization(trainer, env_config, render=True, num_episodes=1):
    """Run visualization with trained model.
    
    Args:
        trainer: PPOTrainer with loaded policies
        env_config: Environment configuration
        render: Whether to show SUMO GUI
        num_episodes: Number of episodes to run
    """
    # Create environment with rendering enabled
    # Need to create a new network for the visualization environment
    network = HighwayNetwork(
        name="highway",
        vehicles=flow_params["veh"],
        net_params=flow_params["net"],
        initial_config=flow_params["initial"],
    )
    
    env_config_copy = {
        "env_params": flow_params["env"],
        "sim_params": flow_params["sim"],
        "network": network,
        "simulator": "traci",
    }
    env_config_copy["sim_params"].render = render
    
    env = MultiAgentPlatoonEnv(env_config_copy)
    
    print("=" * 60)
    print("Starting Visualization")
    print("=" * 60)
    print(f"RL vehicles (red): {env.num_agents}")
    print(f"Human vehicles (blue): {flow_params['veh'].num_vehicles - env.num_agents}")
    print(f"Render mode: {'ON' if render else 'OFF'}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)
    
    if render:
        print("\nSUMO GUI should open shortly...")
        print("RL vehicles are shown in RED")
        print("Human vehicles are shown in BLUE")
        print("\nPress Ctrl+C to stop the simulation\n")
    
    total_rewards = []
    
    try:
        for episode in range(num_episodes):
            obs_dict = env.reset()
            done_dict = {"__all__": False}
            episode_reward = {agent_id: 0.0 for agent_id in env.agent_ids}
            
            step = 0
            while not done_dict["__all__"]:
                # Get actions from trained policies
                action_dict = {}
                for agent_id, obs in obs_dict.items():
                    action = trainer.compute_action(obs, policy_id=agent_id)
                    action_dict[agent_id] = action
                
                # Step environment
                obs_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
                
                # Accumulate rewards
                for agent_id in env.agent_ids:
                    episode_reward[agent_id] += reward_dict.get(agent_id, 0.0)
                
                step += 1
                
                # Print progress every 100 steps
                if step % 100 == 0:
                    avg_reward = np.mean([episode_reward[aid] for aid in env.agent_ids])
                    print(f"Episode {episode+1}, Step {step}: Avg reward = {avg_reward:.2f}")
            
            avg_episode_reward = np.mean([episode_reward[aid] for aid in env.agent_ids])
            total_rewards.append(avg_episode_reward)
            print(f"\nEpisode {episode+1} completed!")
            print(f"Average reward per agent: {avg_episode_reward:.2f}")
            print(f"Total steps: {step}\n")
    
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    
    finally:
        env.env.terminate()
        print("\nSimulation ended")
        if total_rewards:
            print(f"Average reward across episodes: {np.mean(total_rewards):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize independently trained RL cars with human cars in SUMO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint directory or trial directory. If not specified, uses latest checkpoint."
    )
    
    parser.add_argument(
        "--checkpoint_num",
        type=int,
        default=None,
        help="Checkpoint number to load (e.g., 30). If not specified, uses latest checkpoint."
    )
    
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Run without GUI (headless mode)"
    )
    
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    
    try:
        # Find or use checkpoint
        if args.checkpoint_path is None:
            checkpoint_path, checkpoint_num = find_latest_checkpoint()
            if checkpoint_path is None:
                print("ERROR: No checkpoints found in results/multi_agent/")
                print("Please train a model first using train_multi_agent.py")
                return
            print(f"Using latest checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = args.checkpoint_path
            checkpoint_num = args.checkpoint_num
        
        # Load trained model
        trainer, env_config = load_trained_model(checkpoint_path, checkpoint_num)
        
        # Run visualization
        run_visualization(
            trainer, 
            env_config, 
            render=not args.no_render,
            num_episodes=args.num_episodes
        )
    
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

