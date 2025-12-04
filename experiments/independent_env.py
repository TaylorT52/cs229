"""Independent learning environment for platooning.

This environment extends PlatoonEnv to provide per-agent rewards,
making it suitable for truly independent PPO training where each
agent only receives feedback based on its own behavior.
"""

import numpy as np
from platoon_env import PlatoonEnv


class IndependentPlatoonEnv(PlatoonEnv):
    """Environment for independent multi-agent learning.
    
    Key difference from PlatoonEnv:
    - compute_reward() returns a dict of per-agent rewards
    - Each agent's reward depends only on its own actions/state
    - No shared reward signal across agents
    """
    
    def __init__(self, env_params, sim_params, network, simulator="traci"):
        super().__init__(env_params, sim_params, network, simulator)
        self.prev_lanes = {}  # Track previous lanes for lane change detection

    def reset(self):
        """Clear per-episode buffers and delegate to the base reset."""
        self.prev_lanes.clear()
        return super().reset()

    def compute_reward(self, rl_actions, **kwargs):
        """Compute individual rewards for each RL vehicle.

        Rewards are designed to work in multi-lane scenarios where vehicles
        may not always have a same-lane leader. Uses cross-lane headway.
        """
        rl_ids = self.k.vehicle.get_rl_ids()
        if len(rl_ids) == 0:
            return {}

        rewards = {}

        for rl_id in rl_ids:
            speed = self.k.vehicle.get_speed(rl_id)
            headway = self._safe_headway(rl_id)  # Now uses cross-lane headway

            # Start with small baseline reward for staying in simulation
            agent_reward = 0.1

            # 1. Speed reward: encourage driving near target speed
            # Normalized to [-1, 0] range when below target
            speed_ratio = speed / max(self.target_speed, 1e-3)
            if speed_ratio > 1.0:
                speed_reward = -0.5 * (speed_ratio - 1.0)  # Penalty for speeding
            else:
                speed_reward = speed_ratio - 1.0  # -1 at stopped, 0 at target
            agent_reward += 0.3 * speed_reward

            # 2. Following distance reward (works across lanes now)
            if headway is not None:
                if 10.0 <= headway <= 30.0:
                    # Optimal platoon distance - positive reward
                    platoon_reward = 1.0
                elif 30.0 < headway <= 60.0:
                    # Slightly too far but still reasonable - small penalty
                    platoon_reward = 0.5 - 0.01 * (headway - 30.0)
                elif headway > 60.0:
                    # Too far - but cap penalty to prevent domination
                    platoon_reward = max(-0.3, 0.2 - 0.01 * (headway - 60.0))
                elif 5.0 <= headway < 10.0:
                    # Slightly too close
                    platoon_reward = 0.5 * (headway - 5.0) / 5.0
                else:
                    # Very close - handled by safety
                    platoon_reward = 0.0
            else:
                platoon_reward = 0.0
            agent_reward += 0.3 * platoon_reward

            # 3. Stability reward: penalize large speed changes
            if rl_id in self.prev_speeds:
                prev_speed = self.prev_speeds[rl_id]
                speed_change = abs(speed - prev_speed)
                stability_penalty = -min(0.5, speed_change / 5.0)
                agent_reward += 0.2 * stability_penalty
            self.prev_speeds[rl_id] = speed

            # 4. Safety penalty (capped to prevent domination)
            if headway is not None and headway < 5.0:
                safety_penalty = -min(1.0, (5.0 - headway) / 5.0)
                agent_reward += 0.15 * safety_penalty
            elif headway is not None and headway < 8.0:
                safety_penalty = -0.2 * (8.0 - headway) / 3.0
                agent_reward += 0.15 * safety_penalty

            # 5. Lane change penalty (discourage unnecessary lane changes)
            current_lane = self.k.vehicle.get_lane(rl_id)
            if rl_id in self.prev_lanes:
                prev_lane = self.prev_lanes[rl_id]
                if current_lane != prev_lane:
                    # Penalty for lane change - discourage erratic behavior
                    agent_reward += -0.3  # Small penalty per lane change
            self.prev_lanes[rl_id] = current_lane

            rewards[rl_id] = agent_reward

        return rewards

