""" PPO environment for platooning """

from flow.controllers import IDMController, RLController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.envs.base import Env
from flow.networks import HighwayNetwork
import gym
import numpy as np

class PlatoonEnv(Env):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        # Track previous speeds for computing speed variance penalty
        self.prev_speeds = {}
        # Number of observation features per RL vehicle
        self.num_features = 5  # speed, headway, relative_speed, lane, normalized_speed

    @property   
    def action_space(self):
        # Continuous acceleration/deceleration for each RL vehicle
        # Range: -3 to 3 m/s^2 (realistic acceleration limits)
        num_rl = self.initial_vehicles.num_rl_vehicles
        return gym.spaces.Box(
            low=-3.0, 
            high=3.0, 
            shape=(num_rl,), 
            dtype=np.float32
        )
    
    @property
    def observation_space(self):
        # Each RL vehicle observes: speed, headway, relative_speed, lane, normalized_speed
        num_rl = self.initial_vehicles.num_rl_vehicles
        return gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(num_rl * self.num_features,), 
            dtype=np.float32
        )
    
    def get_state(self):
        """Get observations for all RL vehicles.
        
        Each RL vehicle observes:
        - speed: current speed (m/s)
        - headway: distance to vehicle in front (m)
        - relative_speed: speed difference with leader (m/s)
        - lane: current lane number
        - normalized_speed: speed / speed_limit (for better learning)
        """
        states = []
        rl_ids = sorted(self.k.vehicle.get_rl_ids())  # Sort for consistency
        
        for rl_id in rl_ids:
            # Current speed
            speed = self.k.vehicle.get_speed(rl_id)
            
            # Headway (distance to front vehicle) - CORRECT Flow API
            headway = self.k.vehicle.get_headway(rl_id)
            
            # Relative speed (your speed - leader speed)
            leader = self.k.vehicle.get_leader(rl_id)
            if leader is not None and leader != "":
                leader_speed = self.k.vehicle.get_speed(leader)
                relative_speed = speed - leader_speed
            else:
                relative_speed = 0.0  # No leader (first vehicle)
            
            # Current lane
            lane = self.k.vehicle.get_lane(rl_id)
            
            # Normalized speed
            normalized_speed = speed / 30.0  # 30 m/s is speed limit from single_car.py
            
            # Add all features to state
            states.extend([speed, headway, relative_speed, lane, normalized_speed])
        
        return np.array(states, dtype=np.float32)
    
    def compute_reward(self, rl_actions, **kwargs):
        """Compute reward for platooning behavior.
        
        Reward components:
        1. Speed maintenance: reward for staying close to speed limit
        2. Tight platoons: reward for maintaining 10-30m following distance
        3. Stability: penalty for large speed changes (fluctuations)
        4. Safety: large penalty for very small headways (< 5m, collision risk)
        """
        rl_ids = self.k.vehicle.get_rl_ids()
        if len(rl_ids) == 0:
            return 0.0
        
        total_reward = 0.0
        target_speed = 30.0  # Speed limit - WE NEED TO DEFINE THIS
        
        for rl_id in rl_ids:
            speed = self.k.vehicle.get_speed(rl_id)
            headway = self.k.vehicle.get_headway(rl_id)
            
            # Speed maintenance reward (maximize traffic flow)
            # Reward for being close to speed limit
            speed_reward = -abs(speed - target_speed) / target_speed
            total_reward += speed_reward * 0.3  # Weight: 0.3
            
            # Platooning reward (encourage tight following)
            # Optimal platoon distance: 10-30 meters
            if 10.0 <= headway <= 30.0:
                # Give positive reward for good following distance
                platoon_reward = 1.0
            elif headway > 30.0:
                # Penalty for being too far (not platooning)
                platoon_reward = -0.5 * (headway - 30.0) / 100.0
            else:
                # Penalty for being too close (handled by safety below)
                platoon_reward = 0.0
            
            total_reward += platoon_reward * 0.4  # Weight: 0.4
            
            # Stability penalty (penalize speed fluctuations/sudden braking)
            if rl_id in self.prev_speeds:
                prev_speed = self.prev_speeds[rl_id]
                speed_change = abs(speed - prev_speed)
                # Penalize large speed changes (> 2 m/s per step)
                if speed_change > 2.0:
                    stability_penalty = -speed_change / 10.0
                    total_reward += stability_penalty * 0.2  # Weight: 0.2
            
            # Update previous speed
            self.prev_speeds[rl_id] = speed
            
            # Safety penalty (critical for collision avoidance)
            if headway < 5.0:
                # Large penalty for very small headways
                safety_penalty = -10.0 * (5.0 - headway)
                total_reward += safety_penalty * 0.1  # Weight: 0.1
            elif headway < 8.0:
                # Moderately dangerous headways
                safety_penalty = -2.0 * (8.0 - headway)
                total_reward += safety_penalty * 0.1
        
        # Average reward across all RL vehicles
        avg_reward = total_reward / len(rl_ids)
        return avg_reward
    
    def _apply_rl_actions(self, rl_actions):
        """Apply acceleration actions to RL vehicles.
        
        This is called automatically by the parent Env during step().
        """
        if rl_actions is None:
            return
        
        rl_ids = sorted(self.k.vehicle.get_rl_ids())
        if len(rl_ids) == 0:
            return
        
        # Clip actions to safe range (already constrained by action_space, but double-check)
        clipped_actions = np.clip(rl_actions, -3.0, 3.0)
        
        # Apply acceleration to each RL vehicle
        self.k.vehicle.apply_acceleration(rl_ids, clipped_actions)
    
    def additional_command(self):
        """Additional commands to run each simulation step.
        
        Sets RL vehicle colors to red for visual distinction.
        """
        for rl_id in self.k.vehicle.get_rl_ids():
            self.k.kernel_api.vehicle.setColor(rl_id, (255, 0, 0, 255))