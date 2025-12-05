import numpy as np
from gym.spaces import Box, Dict
from ray.rllib.env.multi_agent_env import MultiAgentEnv as RLlibMultiAgentEnv
from independent_env import IndependentPlatoonEnv

class MultiAgentPlatoonEnv(RLlibMultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        
        self.env = IndependentPlatoonEnv(
            env_params=env_config['env_params'],
            sim_params=env_config['sim_params'],
            network=env_config['network'],
            simulator=env_config.get('simulator', 'traci')
        )

        self.num_agents = self.env.initial_vehicles.num_rl_vehicles
        self.num_features = self.env.num_features
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        self.lane_change_enabled = env_config['env_params'].additional_params.get("lane_change_enabled", False)
        self.use_ctde_obs = env_config.get("use_ctde_obs", False)

        if self.use_ctde_obs:
            self._obs_space = Dict({
                "local": Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(self.num_features,),
                    dtype=np.float32
                ),
                "global": Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(self.num_agents * self.num_features,),
                    dtype=np.float32
                ),
            })
        else:
            self._obs_space = Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.num_features,),
                dtype=np.float32
            )
    
        if self.lane_change_enabled:
            self._action_space = Box(
                low=np.array([-3.0, -1.0], dtype=np.float32),
                high=np.array([3.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            self._action_space = Box(
                low=-3.0, 
                high=3.0,
                shape=(1,),
                dtype=np.float32
            )
    
    def reset(self):
        flat_obs = self.env.reset()
        obs_dict = self._split_observations(flat_obs)
        
        return obs_dict
    
    def step(self, action_dict):
        flat_actions = self._flatten_actions(action_dict)

        flat_obs, flat_reward, done, info = self.env.step(flat_actions)
        obs_dict = self._split_observations(flat_obs)
        reward_dict = self._split_rewards(flat_reward)
        done_dict = {agent_id: done for agent_id in self.agent_ids}
        done_dict["__all__"] = done
        info_dict = {agent_id: {} for agent_id in self.agent_ids}
        
        return obs_dict, reward_dict, done_dict, info_dict
    
    def _split_observations(self, flat_obs):
        obs_dict = {}
        global_obs = flat_obs.copy()
        
        for i, agent_id in enumerate(self.agent_ids):
            start_idx = i * self.num_features
            end_idx = start_idx + self.num_features
            agent_obs = flat_obs[start_idx:end_idx]
            
            if self.use_ctde_obs:
                obs_dict[agent_id] = {
                    "local": agent_obs,
                    "global": global_obs
                }
            else:
                obs_dict[agent_id] = agent_obs
        
        return obs_dict
    
    def _flatten_actions(self, action_dict):
        if self.lane_change_enabled:
            flat_actions = np.zeros(self.num_agents * 2, dtype=np.float32)
            
            for i, agent_id in enumerate(self.agent_ids):
                if agent_id in action_dict:
                    action = action_dict[agent_id]
                    if isinstance(action, (list, np.ndarray)):
                        if len(action) >= 2:
                            flat_actions[i * 2] = action[0]
                            flat_actions[i * 2 + 1] = action[1]
                        elif len(action) == 1:
                            flat_actions[i * 2] = action[0]
                            flat_actions[i * 2 + 1] = 0.0
                    else:
                        flat_actions[i * 2] = float(action)
                        flat_actions[i * 2 + 1] = 0.0
        else:
            flat_actions = np.zeros(self.num_agents, dtype=np.float32)
            
            for i, agent_id in enumerate(self.agent_ids):
                if agent_id in action_dict:
                    action = action_dict[agent_id]
                    if isinstance(action, (list, np.ndarray)):
                        action = action[0]
                    flat_actions[i] = action
        
        return flat_actions
    
    def _split_rewards(self, reward_data):
        if isinstance(reward_data, dict):
            rl_ids = sorted(self.env.k.vehicle.get_rl_ids())
            reward_dict = {}
            for i, agent_id in enumerate(self.agent_ids):
                if i < len(rl_ids):
                    veh_id = rl_ids[i]
                    reward_dict[agent_id] = reward_data.get(veh_id, 0.0)
                else:
                    reward_dict[agent_id] = 0.0
            
            return reward_dict
        else:
            reward_dict = {agent_id: reward_data for agent_id in self.agent_ids}
            return reward_dict
    
    def observation_space_sample(self, agent_ids=None):
        if agent_ids is None:
            agent_ids = self.agent_ids
        return {agent_id: self._obs_space for agent_id in agent_ids}
    
    def action_space_sample(self, agent_ids=None):
        if agent_ids is None:
            agent_ids = self.agent_ids
        return {agent_id: self._action_space for agent_id in agent_ids}
    
    def observation_space_contains(self, x):
        return all(self._obs_space.contains(obs) for obs in x.values())
    
    def action_space_contains(self, x):
        return all(self._action_space.contains(act) for act in x.values())
    
    @property
    def observation_space(self):
        return self._obs_space
    
    @property
    def action_space(self):
        return self._action_space
