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
from flow.core.params import SumoCarFollowingParams
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


def run_visualization(trainer, env_config, render=True, num_episodes=1, horizon=None):
    """Run visualization with trained model.
    
    Args:
        trainer: PPOTrainer with loaded policies
        env_config: Environment configuration
        render: Whether to show SUMO GUI
        num_episodes: Number of episodes to run
    """
    # Create a copy of flow_params with realistic parameters for visualization
    from copy import deepcopy
    from flow.controllers import IDMController, RLController, ContinuousRouter
    from flow.core.params import VehicleParams
    
    vis_flow_params = deepcopy(flow_params)
    
    if horizon is None:
        horizon = 10000  # Much longer than default 1500
    vis_flow_params["env"].horizon = horizon

    original_speed_limit = flow_params["net"].additional_params.get("speed_limit", 30.0)
    speed_limit = 15.0  
 
    vis_flow_params["net"].additional_params["speed_limit"] = speed_limit
    
    human_cf_params = SumoCarFollowingParams(
        accel=1.8,          # Moderate acceleration: 1.8 m/s² (realistic for normal driving)
        decel=4.5,          # Realistic deceleration: 4.5 m/s² (comfortable braking)
        sigma=0.2,           # Moderate imperfection: 0.2 (realistic driver variation)
        tau=1.0,            # Reaction time: 1.0s (typical human reaction)
        min_gap=4.0,        # Minimum gap: 4.0m (comfortable following distance - prevents collisions)
        max_speed=speed_limit,  # Enforce speed limit
        speed_factor=0.95,  # Slightly below speed limit (realistic)
        speed_dev=0.05,     # 5% speed variation (normal traffic variation)
        impatience=0.3,     # Moderate impatience: 0.3 (normal driving)
        car_follow_model="IDM",
        speed_mode="obey_safe_speed",  # Obey safe speed - allows natural car-following
    )
    
    rl_cf_params = SumoCarFollowingParams(
        accel=3.0,          # Better acceleration for autonomous
        decel=5.0,          # Better deceleration
        sigma=0.1,          # Lower imperfection
        tau=0.5,            # Faster reaction: 0.5s
        min_gap=2.0,        # Tighter gap
        max_speed=speed_limit,  # Enforce speed limit
        speed_factor=0.95,  # Slightly below speed limit
        speed_dev=0.05,     # 5% speed variation
        impatience=0.3,
        car_follow_model="IDM",
        speed_mode="all_checks",  # Use all safety checks
    )
    

    vis_vehicles = VehicleParams()
    vis_vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "v0": speed_limit * 0.95,  # Desired speed slightly below limit (realistic)
            "T": 1.8,                  # Safe time headway: 1.8s (comfortable following - prevents tailgating)
            "a": 1.8,                  # Max acceleration: 1.8 m/s² (moderate)
            "b": 4.5,                  # Comfortable deceleration: 4.5 m/s² (matches SumoCarFollowingParams)
            "delta": 4,                 # Acceleration exponent (standard)
            "s0": 4.0,                  # Minimum gap: 4.0m (safe distance - prevents collisions)
            "noise": 0.05,              # Small noise for natural variation
        }),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=human_cf_params,
        num_vehicles=flow_params["veh"].num_vehicles - flow_params["veh"].num_rl_vehicles,
        color="0,100,255",
    )
    vis_vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=rl_cf_params,
        num_vehicles=flow_params["veh"].num_rl_vehicles,
        color="255,0,0",
    )
    
    # Create environment with rendering enabled
    # Need to create a new network for the visualization environment
    network = HighwayNetwork(
        name="highway",
        vehicles=vis_vehicles,
        net_params=vis_flow_params["net"],
        initial_config=vis_flow_params["initial"],
    )
    
    # Ensure the horizon is properly set in env_params
    # Create a fresh EnvParams with the updated horizon
    from flow.core.params import EnvParams
    updated_env_params = EnvParams(
        horizon=vis_flow_params["env"].horizon,
        warmup_steps=vis_flow_params["env"].warmup_steps if hasattr(vis_flow_params["env"], 'warmup_steps') else 0,
        sims_per_step=vis_flow_params["env"].sims_per_step if hasattr(vis_flow_params["env"], 'sims_per_step') else 1,
        additional_params=vis_flow_params["env"].additional_params if hasattr(vis_flow_params["env"], 'additional_params') else {},
    )
    
    env_config_copy = {
        "env_params": updated_env_params,
        "sim_params": vis_flow_params["sim"],
        "network": network,
        "simulator": "traci",
    }
    env_config_copy["sim_params"].render = render
    
    print(f"Environment horizon set to: {updated_env_params.horizon} steps ({updated_env_params.horizon * 0.1:.1f} seconds)")
    
    env = MultiAgentPlatoonEnv(env_config_copy)
    
    # Disable collision teleportation - keep vehicles at collision locations
    # Set collision.action to "none" so vehicles stay where they collide
    try:
        traci = env.env.k.kernel_api
        # Set collision action to "none" - vehicles will remain at collision location
        # Options: "none" (keep vehicles), "warn" (warn but keep), "teleport" (default), "remove"
        traci.simulation.setParameter("", "collision.action", "none")
        print("Collision teleportation disabled - vehicles will remain at collision locations")
    except Exception as e:
        print(f"Warning: Could not set collision parameters: {e}")
        print("Attempting alternative method...")
        try:
            # Alternative: set via sumoParams if available
            if hasattr(env.env.k.sim_params, 'additional_params'):
                env.env.k.sim_params.additional_params['collision.action'] = 'none'
        except:
            pass
    
    simulation_duration = vis_flow_params["env"].horizon * 0.1  # Convert steps to seconds
    
    print("=" * 60)
    print("Starting Visualization")
    print("=" * 60)
    print(f"RL vehicles (red): {env.num_agents}")
    print(f"Human vehicles (blue): {vis_vehicles.num_vehicles - env.num_agents}")
    print(f"Render mode: {'ON' if render else 'OFF'}")
    print(f"Episodes: {num_episodes}")
    print(f"Simulation duration: {simulation_duration:.1f} seconds ({simulation_duration/60:.1f} minutes) per episode")
    print("=" * 60)
    
    if render:
        print("\nSUMO GUI should open shortly...")
        print("RL vehicles are shown in RED")
        print("Human vehicles are shown in BLUE")
        print("Vehicles will wrap around when they reach the end")
        print("\nPress Ctrl+C to stop the simulation\n")
    
    total_rewards = []
    
    try:
        for episode in range(num_episodes):
            obs_dict = env.reset()
            done_dict = {"__all__": False}
            episode_reward = {agent_id: 0.0 for agent_id in env.agent_ids}
            
            # Set vehicle colors explicitly for visibility
            rl_ids = env.env.k.vehicle.get_rl_ids()
            for rl_id in rl_ids:
                env.env.k.kernel_api.vehicle.setColor(rl_id, (255, 0, 0, 255))  # Red for RL
            
            human_ids = env.env.k.vehicle.get_human_ids()
            for human_id in human_ids:
                env.env.k.kernel_api.vehicle.setColor(human_id, (0, 100, 255, 255))  # Blue for human
            
            # Get highway length for wrap-around
            highway_length = vis_flow_params["net"].additional_params.get("length", 1000)
            
            max_steps = vis_flow_params["env"].horizon
            step = 0
            
            while step < max_steps:
                # Get actions from trained policies
                action_dict = {}
                for agent_id, obs in obs_dict.items():
                    action = trainer.compute_action(obs, policy_id=agent_id)
                    action_dict[agent_id] = action
                
                # Step environment
                obs_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
                
                # Check if environment says it's done, but continue anyway for visualization
                # (unless it's a crash or user interruption)
                if done_dict.get("__all__", False):
                    # Check if it's just the horizon being reached
                    # If so, we can continue by resetting or ignoring the done flag
                    # For visualization, we want to run for the full duration
                    if step < max_steps - 100:  # If we're not near the end, something else happened
                        print(f"Warning: Environment signaled done at step {step}, but continuing...")
                    # Reset done flag to continue simulation
                    done_dict["__all__"] = False
                
                # Get speed limit
                speed_limit = vis_flow_params["net"].additional_params.get("speed_limit", 30.0)
                
                # Enforce speed limits and wrap vehicles around
                for veh_id in env.env.k.vehicle.get_ids():
                    try:
                        pos = env.env.k.vehicle.get_position(veh_id)
                        edge = env.env.k.vehicle.get_edge(veh_id)
                        speed = env.env.k.vehicle.get_speed(veh_id)
                        
                        # Only enforce speed limits on human vehicles (blue cars) - but be less aggressive
                        # Let IDM controller do most of the work for natural behavior
                        is_human = veh_id in env.env.k.vehicle.get_human_ids()
                        
                        if is_human:
                            # Only cap speed if it's way over the limit (let IDM handle normal behavior)
                            # This allows natural acceleration/deceleration patterns
                            if speed > speed_limit * 1.1:  # Only cap if 10% over limit
                                try:
                                    # Cap speed at the limit
                                    env.env.k.kernel_api.vehicle.setSpeed(veh_id, speed_limit)
                                except:
                                    pass
                            
                            # Minimal intervention - let IDM controller handle car-following naturally
                            # Only intervene in extreme cases to prevent crashes
                            try:
                                leader = env.env.k.vehicle.get_leader(veh_id)
                                if leader:
                                    headway = env.env.k.vehicle.get_headway(veh_id)
                                    if headway is not None and headway < 2.0:  # Only if dangerously close
                                        leader_speed = env.env.k.vehicle.get_speed(leader)
                                        # Emergency slow down only if very close and much faster
                                        if speed > leader_speed + 2.0:
                                            try:
                                                # Emergency brake to prevent collision
                                                target_speed = max(0, leader_speed - 1.0)
                                                env.env.k.kernel_api.vehicle.setSpeed(veh_id, target_speed)
                                            except:
                                                pass
                            except:
                                pass
                        
                        # Wrap vehicles around: teleport vehicles before they reach the end
                        # Use smarter logic to prevent collisions
                        if "highway" in edge:
                            # Wrap when vehicle is within 100m of the end (even earlier to prevent piling)
                            if pos >= highway_length - 100.0:
                                lane = env.env.k.vehicle.get_lane(veh_id)
                                
                                # Find a safe position at the start of the highway
                                # Check for vehicles already at the start to avoid collisions
                                safe_start_pos = 20.0  # Start 20m from beginning
                                min_safe_distance = 15.0  # Minimum distance between vehicles
                                
                                # Check positions of vehicles on the same lane near the start
                                vehicles_on_lane = []
                                for other_veh_id in env.env.k.vehicle.get_ids():
                                    try:
                                        other_edge = env.env.k.vehicle.get_edge(other_veh_id)
                                        other_pos = env.env.k.vehicle.get_position(other_veh_id)
                                        other_lane = env.env.k.vehicle.get_lane(other_veh_id)
                                        
                                        # If vehicle is on same lane and near the start
                                        if ("highway" in other_edge and 
                                            int(other_lane) == int(lane) and 
                                            other_pos < 200.0 and
                                            other_veh_id != veh_id):
                                            vehicles_on_lane.append(other_pos)
                                    except:
                                        pass
                                
                                # Find a safe position that doesn't overlap with existing vehicles
                                if vehicles_on_lane:
                                    vehicles_on_lane.sort()
                                    # Start checking from safe_start_pos
                                    candidate_pos = safe_start_pos
                                    for existing_pos in vehicles_on_lane:
                                        if abs(candidate_pos - existing_pos) < min_safe_distance:
                                            # Too close, move further
                                            candidate_pos = existing_pos + min_safe_distance
                                    new_pos = candidate_pos
                                else:
                                    # No vehicles ahead, use safe start position
                                    new_pos = safe_start_pos
                                
                                # Ensure we don't go too far
                                if new_pos > highway_length - 200.0:
                                    new_pos = safe_start_pos
                                
                                # Try to move vehicle to the start of the highway
                                try:
                                    # Use moveTo with keepRoute to prevent removal
                                    env.env.k.kernel_api.vehicle.moveTo(
                                        veh_id,
                                        "highway_0",
                                        pos=new_pos,
                                        lane=int(lane),
                                        keepRoute=1  # Keep route to prevent removal
                                    )
                                except:
                                    try:
                                        # Fallback: use moveToXY
                                        env.env.k.kernel_api.vehicle.moveToXY(
                                            veh_id,
                                            edgeID="highway_0",
                                            lane=int(lane),
                                            x=new_pos,
                                            y=0,
                                            angle=0,
                                            keepRoute=1  # Keep route
                                        )
                                    except:
                                        # If both fail, skip this vehicle (don't force teleport)
                                        pass
                    except Exception as e:
                        # If vehicle info not available, continue
                        pass
                
                # Re-apply colors periodically (in case vehicles are added/removed)
                if step % 10 == 0:
                    rl_ids = env.env.k.vehicle.get_rl_ids()
                    for rl_id in rl_ids:
                        env.env.k.kernel_api.vehicle.setColor(rl_id, (255, 0, 0, 255))
                    human_ids = env.env.k.vehicle.get_human_ids()
                    for human_id in human_ids:
                        env.env.k.kernel_api.vehicle.setColor(human_id, (0, 100, 255, 255))
                
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
    
    parser.add_argument(
        "--horizon",
        type=int,
        default=10000,
        help="Simulation horizon in steps (default: 10000 = 1000 seconds = ~16.7 minutes). Each step is 0.1 seconds."
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
            num_episodes=args.num_episodes,
            horizon=args.horizon
        )
    
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

