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
    training_dirs = glob.glob(os.path.join(RESULTS_DIR, "PPO_*"))
    if not training_dirs:
        raise FileNotFoundError("No training results found in results/multi_agent/")
    
    latest_dir = max(training_dirs, key=os.path.getctime)
    trial_dirs = [d for d in os.listdir(latest_dir) 
                  if os.path.isdir(os.path.join(latest_dir, d)) and d.startswith("PPO_")]
    
    if not trial_dirs:
        raise FileNotFoundError(f"No trial directory found in {latest_dir}")
    
    trial_dir = os.path.join(latest_dir, trial_dirs[0])

    params_path = os.path.join(trial_dir, "params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.json not found in {trial_dir}")
    
    with open(params_path, 'r') as f:
        params = json.load(f)

    checkpoint_dirs = glob.glob(os.path.join(trial_dir, "checkpoint_*"))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {trial_dir}")
    
    latest_checkpoint_dir = max(checkpoint_dirs, 
                                 key=lambda x: int(x.split("_")[-1]))
    
    checkpoint_num = int(latest_checkpoint_dir.split("_")[-1])
    checkpoint_path = os.path.join(latest_checkpoint_dir, f"checkpoint-{checkpoint_num}")
    
    return checkpoint_path, params


def create_env_config(render=True):
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
    checkpoint_path, saved_params = find_latest_checkpoint_and_params()
    
    model_config = saved_params.get("model", {})
    multiagent_config = saved_params.get("multiagent", {})
    horizon = saved_params.get("horizon", 1500)
    framework = saved_params.get("framework", "torch")
    
    print(f"\nLoaded config from params.json:")
    print(f"  - Model: fcnet_hiddens={model_config.get('fcnet_hiddens')}")
    print(f"  - Activation: {model_config.get('fcnet_activation')}")
    print(f"  - Horizon: {horizon}")
    print(f"  - Framework: {framework}")
    
    saved_policies = multiagent_config.get("policies", {})
    num_agents = len(saved_policies)
    print(f"  - Number of agents: {num_agents}")
    
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    
    def env_creator(env_config):
        return MultiAgentPlatoonEnv(env_config)
    register_env("platoon_multiagent", env_creator)

    obs_space = Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
    act_space = Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
    
    policies = {
        f"agent_{i}": (None, obs_space, act_space, {})
        for i in range(num_agents)
    }
    
    def policy_mapping_fn(agent_id, **kwargs):
        return agent_id
    
    config = ppo.DEFAULT_CONFIG.copy()
    config.update({
        "env": "platoon_multiagent",
        "env_config": create_env_config(render=True),
        "horizon": horizon,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "model": model_config,
        "num_workers": 0,
        "framework": framework,
    })
    
    print("\nLoading trained model...")
    trainer = ppo.PPOTrainer(config=config)
    trainer.restore(checkpoint_path)
    print("Model loaded successfully!")
    
    print("\nStarting simulation...")
    env = MultiAgentPlatoonEnv(create_env_config(render=True))

    num_episodes = 3
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        obs = env.reset()
        done = {"__all__": False}
        total_reward = {f"agent_{i}": 0.0 for i in range(num_agents)}
        step = 0
        
        while not done["__all__"]:
            actions = {}
            for agent_id in obs.keys():
                action = trainer.compute_single_action(
                    obs[agent_id],
                    policy_id=agent_id,
                )
                actions[agent_id] = action
            
            obs, rewards, done, info = env.step(actions)

            for agent_id, reward in rewards.items():
                total_reward[agent_id] += reward
            
            step += 1
            
            if step % 100 == 0:
                avg_reward = np.mean(list(total_reward.values()))
                print(f"  Step {step}: avg reward = {avg_reward:.2f}")
        
        print(f"\nEpisode {episode + 1} complete!")
        print(f"  Total steps: {step}")
        for agent_id, reward in total_reward.items():
            print(f"  {agent_id}: {reward:.2f}")
        print(f"  Average: {np.mean(list(total_reward.values())):.2f}")
    env.env.terminate()
    ray.shutdown()

if __name__ == "__main__":
    main()
