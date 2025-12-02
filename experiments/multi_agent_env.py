"""Multi-agent wrapper for IndependentPlatoonEnv to work with RLlib's multi-agent API.

This wrapper converts IndependentPlatoonEnv (which returns flat observations and 
per-agent reward dicts) into the format expected by RLlib for independent multi-agent learning.
"""

import numpy as np
from gym.spaces import Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv as RLlibMultiAgentEnv

from independent_env import IndependentPlatoonEnv


class MultiAgentPlatoonEnv(RLlibMultiAgentEnv):
    """Wrapper that converts IndependentPlatoonEnv to RLlib's multi-agent format.
    
    This enables independent PPO training where each RL vehicle is treated
    as a separate agent with its own policy and receives individual rewards.
    """
    
    def __init__(self, env_config):
        """Initialize the multi-agent environment.
        
        Args:
            env_config: Dictionary containing Flow parameters:
                - env_params: EnvParams object
                - sim_params: SumoParams object  
                - network: Network object
                - simulator: 'traci' (default)
        """
        super().__init__()
        
        # Create the underlying Flow environment (with per-agent rewards)
        self.env = IndependentPlatoonEnv(
            env_params=env_config['env_params'],
            sim_params=env_config['sim_params'],
            network=env_config['network'],
            simulator=env_config.get('simulator', 'traci')
        )
        
        # Number of RL vehicles (agents)
        self.num_agents = self.env.initial_vehicles.num_rl_vehicles
        self.num_features = 5  # Features per agent
        
        # Create agent IDs: "agent_0", "agent_1", ..., "agent_9"
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        
        # Store observation and action spaces (same for all agents)
        self._obs_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.num_features,),
            dtype=np.float32
        )
        self._action_space = Box(
            low=-3.0, 
            high=3.0, 
            shape=(1,),  # Single continuous action per agent
            dtype=np.float32
        )
    
    def reset(self):
        """Reset the environment and return initial observations.
        
        Returns:
            obs_dict: Dictionary mapping agent_id -> observation array
        """
        # Reset the underlying Flow environment
        flat_obs = self.env.reset()
        
        # Split flat observation into per-agent observations
        obs_dict = self._split_observations(flat_obs)
        
        return obs_dict
    
    def step(self, action_dict):
        """Take a step in the environment.
        
        Args:
            action_dict: Dictionary mapping agent_id -> action
                Example: {"agent_0": [0.5], "agent_1": [-0.3], ...}
        
        Returns:
            obs_dict: Dictionary of observations per agent
            reward_dict: Dictionary of rewards per agent
            done_dict: Dictionary of done flags (+ "__all__" key)
            info_dict: Dictionary of info per agent
        """
        # Convert action dict to flat array for Flow environment
        flat_actions = self._flatten_actions(action_dict)
        
        # Step the underlying Flow environment
        flat_obs, flat_reward, done, info = self.env.step(flat_actions)
        
        # Split observations into per-agent format
        obs_dict = self._split_observations(flat_obs)
        
        # Split rewards into per-agent format
        reward_dict = self._split_rewards(flat_reward)
        
        # Create done dict (all agents done at same time in this env)
        done_dict = {agent_id: done for agent_id in self.agent_ids}
        done_dict["__all__"] = done  # Required by RLlib
        
        # Create info dict (can add per-agent info if needed)
        info_dict = {agent_id: {} for agent_id in self.agent_ids}
        
        return obs_dict, reward_dict, done_dict, info_dict
    
    def _split_observations(self, flat_obs):
        """Split flat observation array into per-agent observations.
        
        Args:
            flat_obs: Flat numpy array of shape (num_agents * num_features,)
                Example: [speed_0, headway_0, ..., speed_1, headway_1, ...]
        
        Returns:
            obs_dict: Dictionary mapping agent_id -> observation array of shape (num_features,)
        """
        obs_dict = {}
        
        for i, agent_id in enumerate(self.agent_ids):
            # Extract the features for this agent
            start_idx = i * self.num_features
            end_idx = start_idx + self.num_features
            agent_obs = flat_obs[start_idx:end_idx]
            
            obs_dict[agent_id] = agent_obs
        
        return obs_dict
    
    def _flatten_actions(self, action_dict):
        """Convert per-agent action dict to flat array for Flow environment.
        
        Args:
            action_dict: Dictionary mapping agent_id -> action
                Example: {"agent_0": [0.5], "agent_1": [-0.3], ...}
        
        Returns:
            flat_actions: Numpy array of shape (num_agents,)
        """
        flat_actions = np.zeros(self.num_agents, dtype=np.float32)
        
        for i, agent_id in enumerate(self.agent_ids):
            if agent_id in action_dict:
                # Extract scalar action (remove array wrapper if present)
                action = action_dict[agent_id]
                if isinstance(action, (list, np.ndarray)):
                    action = action[0]
                flat_actions[i] = action
        
        return flat_actions
    
    def _split_rewards(self, reward_data):
        """Convert Flow vehicle IDs to RLlib agent IDs for rewards.
        
        IndependentPlatoonEnv returns per-agent rewards as a dict with
        vehicle IDs as keys (e.g., "rl_vehicle_0"). We need to map these
        to agent IDs (e.g., "agent_0") for RLlib.
        
        Args:
            reward_data: Dictionary mapping vehicle_id -> reward (from IndependentPlatoonEnv)
                Example: {"rl_vehicle_0": 0.3, "rl_vehicle_1": -0.2, ...}
        
        Returns:
            reward_dict: Dictionary mapping agent_id -> reward
                Example: {"agent_0": 0.3, "agent_1": -0.2, ...}
        """
        # IndependentPlatoonEnv returns dict of per-agent rewards
        if isinstance(reward_data, dict):
            # Get current RL vehicle IDs (sorted for consistency)
            rl_ids = sorted(self.env.k.vehicle.get_rl_ids())
            
            # Map vehicle IDs (Flow) to agent IDs (RLlib)
            reward_dict = {}
            for i, agent_id in enumerate(self.agent_ids):
                if i < len(rl_ids):
                    veh_id = rl_ids[i]
                    # Get this vehicle's individual reward
                    reward_dict[agent_id] = reward_data.get(veh_id, 0.0)
                else:
                    reward_dict[agent_id] = 0.0
            
            return reward_dict
        
        # Handle scalar format (shouldn't happen with IndependentPlatoonEnv)
        else:
            reward_dict = {agent_id: reward_data for agent_id in self.agent_ids}
            return reward_dict
    
    def observation_space_sample(self, agent_ids=None):
        """Return observation space for specified agents."""
        if agent_ids is None:
            agent_ids = self.agent_ids
        return {agent_id: self._obs_space for agent_id in agent_ids}
    
    def action_space_sample(self, agent_ids=None):
        """Return action space for specified agents."""
        if agent_ids is None:
            agent_ids = self.agent_ids
        return {agent_id: self._action_space for agent_id in agent_ids}
    
    def observation_space_contains(self, x):
        """Check if observation is valid."""
        return all(self._obs_space.contains(obs) for obs in x.values())
    
    def action_space_contains(self, x):
        """Check if action is valid."""
        return all(self._action_space.contains(act) for act in x.values())
    
    @property
    def observation_space(self):
        """Return observation space (per agent)."""
        return self._obs_space
    
    @property
    def action_space(self):
        """Return action space (per agent)."""
        return self._action_space
