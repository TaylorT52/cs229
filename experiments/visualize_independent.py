import os
import sys
import argparse
from datetime import datetime
import glob
import json
import numpy as np

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
    if results_dir is None:
        results_dir = os.path.join(project_root, "results", "independent")
    
    if not os.path.exists(results_dir):
        return None, None
    
    exp_dirs = glob.glob(os.path.join(results_dir, "PPO_*"))
    if not exp_dirs:
        return None, None
    
    latest_exp = max(exp_dirs, key=os.path.getmtime)
    trial_dirs = glob.glob(os.path.join(latest_exp, "PPO_*"))
    if not trial_dirs:
        return None, None
    
    latest_trial = max(trial_dirs, key=os.path.getmtime)
    checkpoint_dirs = glob.glob(os.path.join(latest_trial, "checkpoint_*"))
    if not checkpoint_dirs:
        return None, None
    
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
    checkpoint_num = int(os.path.basename(latest_checkpoint).split("_")[1])
    
    return latest_checkpoint, checkpoint_num


def load_trained_model(checkpoint_path, checkpoint_num=None):
    is_checkpoint_dir = os.path.basename(checkpoint_path).startswith("checkpoint_")
    if is_checkpoint_dir:
        checkpoint_dir = checkpoint_path
        checkpoint_num = int(os.path.basename(checkpoint_dir).split("_")[1])
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_num}")
    elif checkpoint_num is not None:
        checkpoint_dir = os.path.join(checkpoint_path, f"checkpoint_{checkpoint_num:06d}")
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_num}")
    else:
        checkpoint_dirs = glob.glob(os.path.join(checkpoint_path, "checkpoint_*"))
        if not checkpoint_dirs:
            raise ValueError(f"No checkpoints found in {checkpoint_path}")
        checkpoint_dir = max(checkpoint_dirs, key=os.path.getmtime)
        checkpoint_num = int(os.path.basename(checkpoint_dir).split("_")[1])
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_num}")
    
    if not os.path.exists(checkpoint_file):
        raise ValueError(f"Checkpoint file not found: {checkpoint_file}")
    
    trial_dir = os.path.dirname(checkpoint_dir) if is_checkpoint_dir else checkpoint_path
    params_file = os.path.join(trial_dir, "params.json")
    
    saved_config = {}
    if os.path.exists(params_file):
        print(f"Loading config from: {params_file}")
        with open(params_file, 'r') as f:
            saved_config = json.load(f)
    else:
        print(f"Warning: params.json not found at {params_file}, using defaults")
    
    def env_creator(env_config):
        return MultiAgentPlatoonEnv(env_config)
    
    register_env("platoon_multiagent", env_creator)
    
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
    
    num_agents = flow_params["veh"].num_rl_vehicles
    
    lane_change_enabled = flow_params["env"].additional_params.get("lane_change_enabled", False)
    if lane_change_enabled:
        num_features = 18
    else:
        num_features = 9
    
    obs_space = Box(
        low=-np.inf, 
        high=np.inf, 
        shape=(num_features,),
        dtype=np.float32
    )

    if lane_change_enabled:
        act_space = Box(
            low=np.array([-3.0, -1.0], dtype=np.float32),
            high=np.array([3.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
    else:
        act_space = Box(
            low=-3.0, 
            high=3.0, 
            shape=(1,),
            dtype=np.float32
        )
    
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

    config = ppo.DEFAULT_CONFIG.copy()
    model_config = saved_config.get("model", {})
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
        "model": model_config,
    })

    if "lr" in saved_config:
        config["lr"] = saved_config["lr"]
    if "gamma" in saved_config:
        config["gamma"] = saved_config["gamma"]
    if "lambda" in saved_config:
        config["lambda"] = saved_config["lambda"]
    if "clip_param" in saved_config:
        config["clip_param"] = saved_config["clip_param"]
    
    trainer = ppo.PPOTrainer(config=config)
    trainer.restore(checkpoint_file)
    
    return trainer, env_config


def run_visualization(trainer, env_config, render=True, num_episodes=1, horizon=None):
    from copy import deepcopy
    from flow.controllers import IDMController, RLController, ContinuousRouter, SimLaneChangeController
    from flow.core.params import VehicleParams
    
    vis_flow_params = deepcopy(flow_params)
    vis_flow_params["env"].additional_params["lane_change_enabled"] = True
    
    if horizon is None:
        horizon = 10000 
    vis_flow_params["env"].horizon = horizon

    original_speed_limit = flow_params["net"].additional_params.get("speed_limit", 30.0)
    speed_limit = 15.0  
    vis_flow_params["net"].additional_params["length"] = 3000 
    vis_flow_params["net"].additional_params["speed_limit"] = speed_limit
    
    human_cf_params = SumoCarFollowingParams(
        accel=1.8,
        decel=4.5,
        sigma=0.2, 
        tau=1.0, 
        min_gap=4.0, 
        max_speed=speed_limit,
        speed_factor=0.95,
        speed_dev=0.05,
        impatience=0.3, 
        car_follow_model="IDM",
        speed_mode="obey_safe_speed",
    )
    
    rl_cf_params = SumoCarFollowingParams(
        accel=3.0,      
        decel=5.0, 
        sigma=0.1, 
        tau=0.5,    
        min_gap=2.0,       
        max_speed=speed_limit, 
        speed_factor=0.95,  
        speed_dev=0.05,   
        impatience=0.3,
        car_follow_model="IDM",
        speed_mode="all_checks", 
    )
    

    from flow.core.params import InFlows
    
    vis_vehicles = VehicleParams()
    vis_vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "v0": speed_limit * 0.95,  
            "T": 1.8,            
            "a": 1.8,           
            "b": 4.5,             
            "delta": 4,               
            "s0": 4.0,            
            "noise": 0.05,              
        }),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=human_cf_params,
        num_vehicles=5,
        color="0,100,255",
    )
    vis_vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        lane_change_controller=(SimLaneChangeController, {}), 
        car_following_params=rl_cf_params,
        num_vehicles=flow_params["veh"].num_rl_vehicles,
        color="255,0,0",
    )
    
    inflows = InFlows()
    inflows.add(
        veh_type="human",
        edge="highway_0",
        vehs_per_hour=1200,  
        depart_speed="max", 
        depart_lane="random",  
        begin=1, 
        end=horizon * 0.1,  
    )

    inflows.add(
        veh_type="rl",
        edge="highway_0",
        vehs_per_hour=360, 
        depart_speed="max",
        depart_lane="random",
        begin=1,
        end=horizon * 0.1,
    )
    
    vis_flow_params["net"].inflows = inflows
    
    network = HighwayNetwork(
        name="highway",
        vehicles=vis_vehicles,
        net_params=vis_flow_params["net"],
        initial_config=vis_flow_params["initial"],
    )
    
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
    try:
        traci = env.env.k.kernel_api
        traci.simulation.setParameter("", "collision.action", "warn")
        traci.simulation.setParameter("", "collision.stoptime", "10000")
        print("Collision teleportation disabled - vehicles will remain at collision locations")
    except Exception as e:
        print(f"Warning: Could not set collision parameters: {e}")
        print("Attempting alternative method...")
        try:
            if hasattr(env.env.k.sim_params, 'additional_params'):
                env.env.k.sim_params.additional_params['collision.action'] = 'warn'
        except:
            pass
    
    simulation_duration = vis_flow_params["env"].horizon * 0.1 
    
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
    print("=" * 60)
    
    if render:
        print("\nSUMO GUI should open shortly...")
        print("RL vehicles are shown in RED")
        print("Human vehicles are shown in BLUE")
        print("Vehicles enter from the left and exit on the right")
        print("Continuous traffic flow - no wraparound artifacts")
        print("\nPress Ctrl+C to stop the simulation\n")
    
    total_rewards = []
    
    metrics_collector = MetricsCollector(policy_type="independent")
    
    try:
        for episode in range(num_episodes):
            obs_dict = env.reset()
            done_dict = {"__all__": False}
            episode_reward = {agent_id: 0.0 for agent_id in env.agent_ids}
            
            if episode > 0:
                metrics_collector = MetricsCollector(policy_type="independent")

            rl_ids = env.env.k.vehicle.get_rl_ids()
            for rl_id in rl_ids:
                env.env.k.kernel_api.vehicle.setColor(rl_id, (255, 0, 0, 255)) 
            
            human_ids = env.env.k.vehicle.get_human_ids()
            for human_id in human_ids:
                env.env.k.kernel_api.vehicle.setColor(human_id, (0, 100, 255, 255))  
            
            highway_length = vis_flow_params["net"].additional_params.get("length", 1000)
            
            max_steps = vis_flow_params["env"].horizon
            step = 0
            
            while step < max_steps:
                action_dict = {}
                for agent_id, obs in obs_dict.items():
                    try:
                        action = trainer.compute_action(obs, policy_id=agent_id)
                        action_dict[agent_id] = action
                    except:
                        pass
                
                obs_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
      
                current_time = step * 0.1 
                metrics_collector.collect_step(env, action_dict, reward_dict, step, current_time)

                if done_dict.get("__all__", False):
                    if step < max_steps - 100:
                        print(f"Warning: Environment signaled done at step {step}, but continuing...")
                    done_dict["__all__"] = False

                speed_limit = vis_flow_params["net"].additional_params.get("speed_limit", 30.0)
  
                for veh_id in env.env.k.vehicle.get_ids():
                    try:
                        pos = env.env.k.vehicle.get_position(veh_id)
                        edge = env.env.k.vehicle.get_edge(veh_id)
                        speed = env.env.k.vehicle.get_speed(veh_id)
                        is_human = veh_id in env.env.k.vehicle.get_human_ids()
                        
                        if is_human:
                            if speed > speed_limit * 1.1: 
                                try:
                                    env.env.k.kernel_api.vehicle.setSpeed(veh_id, speed_limit)
                                except:
                                    pass
                            
                            try:
                                leader = env.env.k.vehicle.get_leader(veh_id)
                                if leader:
                                    headway = env.env.k.vehicle.get_headway(veh_id)
                                    if headway is not None and headway < 2.0: 
                                        leader_speed = env.env.k.vehicle.get_speed(leader)
                                        if speed > leader_speed + 2.0:
                                            try:
                                                target_speed = max(0, leader_speed - 1.0)
                                                env.env.k.kernel_api.vehicle.setSpeed(veh_id, target_speed)
                                            except:
                                                pass
                            except:
                                pass
                        
                    except Exception as e:
                        pass
                
                if step % 10 == 0:
                    rl_ids = env.env.k.vehicle.get_rl_ids()
                    for rl_id in rl_ids:
                        try:
                            env.env.k.kernel_api.vehicle.setColor(rl_id, (255, 0, 0, 255))
                        except:
                            pass
                    human_ids = env.env.k.vehicle.get_human_ids()
                    for human_id in human_ids:
                        try:
                            env.env.k.kernel_api.vehicle.setColor(human_id, (0, 100, 255, 255))
                        except:
                            pass 

                for agent_id in env.agent_ids:
                    episode_reward[agent_id] += reward_dict.get(agent_id, 0.0)
                
                step += 1
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

        print("\n" + "=" * 60)
        print("Computing Metrics...")
        print("=" * 60)
        
        all_metrics = metrics_collector.compute_all_metrics()
        
        # Print summary
        print("\nMetrics Summary:")
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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        metrics_file = os.path.join(metrics_dir, f"independent_metrics_{timestamp}.json")
        raw_data_file = os.path.join(metrics_dir, f"independent_raw_data_{timestamp}.csv")
        
        metrics_collector.save_metrics(metrics_file)
        metrics_collector.save_raw_data(raw_data_file)
        
        print(f"\nMetrics saved to:")
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
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    
    try:
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
        
        trainer, env_config = load_trained_model(checkpoint_path, checkpoint_num)
        
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

