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
    
    def compute_reward(self, rl_actions, **kwargs):
        """Compute individual rewards for each RL vehicle.
        
        REBALANCED reward function to prevent safety penalties from dominating.
        All rewards are normalized to similar scales (-1 to +1 range).
        
        Returns:
            dict: Dictionary mapping vehicle_id -> individual reward
        """
        rl_ids = self.k.vehicle.get_rl_ids()
        if len(rl_ids) == 0:
            return {}

        rewards = {}

        for rl_id in rl_ids:
            speed = self.k.vehicle.get_speed(rl_id)
            headway = self._safe_headway(rl_id)

            # Start with small positive baseline (survival bonus)
            agent_reward = 0.1

            # 1. Speed maintenance reward (normalized to -1 to 0)
            # Target: 30 m/s, reward based on how close we are
            speed_ratio = speed / max(self.target_speed, 1e-3)
            if speed_ratio > 1.0:
                # Over speed limit - small penalty
                speed_reward = -0.5 * (speed_ratio - 1.0)
            else:
                # Under speed limit - reward for being close
                speed_reward = speed_ratio - 1.0  # 0 at target, -1 at stopped
            agent_reward += 0.3 * speed_reward

            # 2. Platooning reward (normalized to -0.5 to +1)
            if 10.0 <= headway <= 30.0:
                # Optimal range - full reward
                platoon_reward = 1.0
            elif headway > 30.0:
                # Too far - gradual penalty (capped)
                platoon_reward = max(-0.5, -0.01 * (headway - 30.0))
            elif headway > 5.0:
                # Slightly too close but safe
                platoon_reward = 0.5 * (headway - 5.0) / 5.0  # 0 at 5m, 0.5 at 10m
            else:
                # Too close - this transitions to safety penalty
                platoon_reward = 0.0
            agent_reward += 0.3 * platoon_reward

            # 3. Stability reward (normalized to -0.5 to 0)
            if rl_id in self.prev_speeds:
                prev_speed = self.prev_speeds[rl_id]
                speed_change = abs(speed - prev_speed)
                # Smooth scaling instead of threshold
                stability_penalty = -min(0.5, speed_change / 5.0)
                agent_reward += 0.2 * stability_penalty
            self.prev_speeds[rl_id] = speed

            # 4. Safety penalty (CAPPED to prevent domination)
            # Max penalty is -1.0 per step (not -5.0!)
            if headway < 5.0:
                # Dangerous - penalty proportional to how close
                safety_penalty = -min(1.0, (5.0 - headway) / 5.0)
                agent_reward += 0.2 * safety_penalty
            elif headway < 8.0:
                # Warning zone - small penalty
                safety_penalty = -0.2 * (8.0 - headway) / 3.0
                agent_reward += 0.2 * safety_penalty

            rewards[rl_id] = agent_reward

        return rewards

