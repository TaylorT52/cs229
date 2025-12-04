import os
import sys
import argparse
from datetime import datetime
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
from metrics_collector import MetricsCollector


def find_latest_checkpoint(results_dir=None):
    """Find the latest checkpoint from training runs.
    
    Args:
        results_dir: Directory to search for checkpoints. If None, uses results/multi_agent
    
    Returns:
        tuple: (checkpoint_dir, checkpoint_num) or (None, None) if not found
    """
    if results_dir is None:
        results_dir = os.path.join(project_root, "results", "independent")
    
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
    
    # Check if lane changes are enabled to determine observation space
    lane_change_enabled = flow_params["env"].additional_params.get("lane_change_enabled", False)
    if lane_change_enabled:
        num_features = 18  # 9 longitudinal + 9 lateral
    else:
        num_features = 9  # 9 longitudinal features only
    
    obs_space = Box(
        low=-np.inf, 
        high=np.inf, 
        shape=(num_features,),
        dtype=np.float32
    )
    
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
    from flow.controllers import IDMController, RLController, ContinuousRouter, SimLaneChangeController
    from flow.core.params import VehicleParams
    
    vis_flow_params = deepcopy(flow_params)
    
    # Enable lane changes for visualization
    vis_flow_params["env"].additional_params["lane_change_enabled"] = True
    
    if horizon is None:
        horizon = 10000  # Much longer than default 1500
    vis_flow_params["env"].horizon = horizon

    original_speed_limit = flow_params["net"].additional_params.get("speed_limit", 30.0)
    speed_limit = 15.0  
 
    # Use longer highway for realistic continuous traffic (recommended approach)
    # This avoids wraparound issues and provides natural inflow/outflow
    vis_flow_params["net"].additional_params["length"] = 3000  # 3km highway
    vis_flow_params["net"].additional_params["speed_limit"] = speed_limit
    
    # Note: Collision parameters are set via TraCI after environment creation
    # (see code below where we disable teleports)
    
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
    

    # Use inflow/outflow for continuous realistic traffic (recommended approach)
    # This avoids wraparound issues and provides natural traffic flow
    from flow.core.params import InFlows
    
    vis_vehicles = VehicleParams()
    # Start with fewer initial vehicles - inflow will add more continuously
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
        num_vehicles=5,  # Fewer initial vehicles - reduced to prevent initial congestion
        color="0,100,255",
    )
    vis_vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        lane_change_controller=(SimLaneChangeController, {}),  # Enable RL-controlled lane changes
        car_following_params=rl_cf_params,
        num_vehicles=flow_params["veh"].num_rl_vehicles,
        color="255,0,0",
    )
    
    # Add continuous inflow of human and RL vehicles for realistic traffic
    # Inflows are passed to NetParams, not to the Network constructor
    inflows = InFlows()
    inflows.add(
        veh_type="human",
        edge="highway_0",
        vehs_per_hour=1200,  # Reduced from 1800 to prevent congestion (~0.33 vehicles per second)
        depart_speed="max",  # Use new parameter name (departSpeed is deprecated)
        depart_lane="random",  # Use new parameter name (departLane is deprecated)
        begin=1,  # Must be >= 1 second (Flow requirement)
        end=horizon * 0.1,  # Continue for entire simulation duration
    )
    # Add RL vehicles to inflow (30% of human flow for mixed traffic)
    inflows.add(
        veh_type="rl",
        edge="highway_0",
        vehs_per_hour=360,  # Reduced from 540 (~0.1 vehicles per second)
        depart_speed="max",
        depart_lane="random",
        begin=1,
        end=horizon * 0.1,
    )
    
    # Add inflows to NetParams
    vis_flow_params["net"].inflows = inflows
    
    # Create environment with rendering enabled
    # Need to create a new network for the visualization environment
    # Use inflow/outflow for continuous realistic traffic
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
    # Set collision.action to "warn" so vehicles stay where they collide (critical for stable RL)
    try:
        traci = env.env.k.kernel_api
        # Set collision action to "warn" - vehicles will remain at collision location
        # Options: "none" (keep vehicles), "warn" (warn but keep), "teleport" (default), "remove"
        traci.simulation.setParameter("", "collision.action", "warn")
        traci.simulation.setParameter("", "collision.stoptime", "10000")
        print("✅ Collision teleportation disabled - vehicles will remain at collision locations")
    except Exception as e:
        print(f"Warning: Could not set collision parameters: {e}")
        print("Attempting alternative method...")
        try:
            # Alternative: set via sumoParams if available
            if hasattr(env.env.k.sim_params, 'additional_params'):
                env.env.k.sim_params.additional_params['collision.action'] = 'warn'
        except:
            pass
    
    simulation_duration = vis_flow_params["env"].horizon * 0.1  # Convert steps to seconds
    
    print("=" * 60)
    print("Starting Visualization - Continuous Traffic Mode")
    print("=" * 60)
    print(f"Initial RL vehicles (red): {env.num_agents}")
    print(f"Initial human vehicles (blue): {vis_vehicles.num_vehicles - env.num_agents}")
    print(f"Human vehicle inflow: 1200 veh/hour (~0.33 veh/sec)")
    print(f"RL vehicle inflow: 360 veh/hour (~0.1 veh/sec, 30% of human flow)")
    print(f"Highway length: {vis_flow_params['net'].additional_params['length']:.0f}m")
    print(f"Render mode: {'ON' if render else 'OFF'}")
    print(f"Episodes: {num_episodes}")
    print(f"Simulation duration: {simulation_duration:.1f} seconds ({simulation_duration/60:.1f} minutes) per episode")
    print("\n✅ Using recommended approach:")
    print("   - Long straight highway (no wraparound)")
    print("   - Continuous vehicle inflow/outflow")
    print("   - Teleports disabled (collision.action=warn)")
    print("   - Modular headway calculation for stability")
    print("=" * 60)
    
    if render:
        print("\nSUMO GUI should open shortly...")
        print("RL vehicles are shown in RED")
        print("Human vehicles are shown in BLUE")
        print("Vehicles enter from the left and exit on the right")
        print("Continuous traffic flow - no wraparound artifacts")
        print("\nPress Ctrl+C to stop the simulation\n")
    
    total_rewards = []
    
    # Initialize metrics collector
    metrics_collector = MetricsCollector(policy_type="independent")
    
    try:
        for episode in range(num_episodes):
            obs_dict = env.reset()
            done_dict = {"__all__": False}
            episode_reward = {agent_id: 0.0 for agent_id in env.agent_ids}
            
            # Reset metrics collector for new episode
            if episode > 0:
                metrics_collector = MetricsCollector(policy_type="independent")
            
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
                # The environment automatically handles new RL vehicles spawned via inflow
                # by mapping them to agent IDs (round-robin if more RL vehicles than agents)
                action_dict = {}
                for agent_id, obs in obs_dict.items():
                    try:
                        action = trainer.compute_action(obs, policy_id=agent_id)
                        action_dict[agent_id] = action
                    except:
                        # If action computation fails, skip this agent
                        pass
                
                # Step environment
                # New RL vehicles from inflow will automatically be included in obs_dict
                obs_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
                
                # Collect metrics for this step
                current_time = step * 0.1  # Simulation time in seconds
                metrics_collector.collect_step(env, action_dict, reward_dict, step, current_time)
                
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
                        
                        # No wraparound needed - vehicles exit naturally at the end
                        # Inflow/outflow provides continuous traffic without wraparound artifacts
                    except Exception as e:
                        # If vehicle info not available, continue
                        pass
                
                # Re-apply colors periodically (in case vehicles are added/removed via inflow)
                if step % 10 == 0:
                    rl_ids = env.env.k.vehicle.get_rl_ids()
                    for rl_id in rl_ids:
                        try:
                            env.env.k.kernel_api.vehicle.setColor(rl_id, (255, 0, 0, 255))
                        except:
                            pass  # Vehicle might have been removed
                    human_ids = env.env.k.vehicle.get_human_ids()
                    for human_id in human_ids:
                        try:
                            env.env.k.kernel_api.vehicle.setColor(human_id, (0, 100, 255, 255))
                        except:
                            pass  # Vehicle might have been removed
                
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
        
        # Compute and save metrics
        print("\n" + "=" * 60)
        print("Computing Metrics...")
        print("=" * 60)
        
        all_metrics = metrics_collector.compute_all_metrics()
        
        # Print summary
        print("\n📊 Metrics Summary:")
        print(f"  Policy Type: {all_metrics['policy_type']}")
        
        spacing = all_metrics.get('spacing_stability', {})
        if spacing:
            print(f"  Avg Spacing Variance: {spacing.get('avg_spacing_variance', 0):.4f}")
            print(f"  Avg Oscillation Amplitude: {spacing.get('avg_oscillation_amplitude', 0):.4f}")
            print(f"  Avg Damping Ratio: {spacing.get('avg_damping_ratio', 0):.4f}")
        
        efficiency = all_metrics.get('efficiency', {})
        if efficiency:
            print(f"  Avg Velocity: {efficiency.get('avg_platoon_velocity', 0):.2f} m/s")
            print(f"  Speed Variance: {efficiency.get('speed_variance', 0):.4f}")
            print(f"  Throughput: {efficiency.get('throughput_vehicles_per_second', 0):.4f} veh/s")
        
        safety = all_metrics.get('safety', {})
        if safety:
            print(f"  Collisions: {safety.get('collision_count', 0)}")
            print(f"  Near Collisions: {safety.get('near_collision_count', 0)}")
            min_ttc = safety.get('min_time_to_collision')
            if min_ttc:
                print(f"  Min TTC: {min_ttc:.2f} s")
        
        coordination = all_metrics.get('coordination', {})
        if coordination:
            print(f"  Action Correlation: {coordination.get('action_correlation', 0):.4f}")
            print(f"  Policy Divergence: {coordination.get('policy_divergence', 0):.4f}")
            print(f"  Synchronization Index: {coordination.get('synchronization_index', 0):.4f}")
        
        string_stab = all_metrics.get('string_stability', {})
        if string_stab:
            print(f"  String Stability Ratio: {string_stab.get('string_stability_ratio', 1.0):.4f}")
            print(f"  Is String Stable: {string_stab.get('is_string_stable', False)}")
        
        # Save metrics to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        metrics_file = os.path.join(metrics_dir, f"independent_metrics_{timestamp}.json")
        raw_data_file = os.path.join(metrics_dir, f"independent_raw_data_{timestamp}.csv")
        
        metrics_collector.save_metrics(metrics_file)
        metrics_collector.save_raw_data(raw_data_file)
        
        print(f"\n✅ Metrics saved to:")
        print(f"   {metrics_file}")
        print(f"   {raw_data_file}")


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
                print("ERROR: No checkpoints found in results/independent/")
                print("Please train a model first using train_independent.py")
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

