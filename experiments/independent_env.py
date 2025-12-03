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

        Rewards mirror the shaping used in PlatoonEnv but are reported
        per-agent so policies can be trained independently.
        """
        rl_ids = self.k.vehicle.get_rl_ids()
        if len(rl_ids) == 0:
            return {}

        rewards = {}

        for rl_id in rl_ids:
            speed = self.k.vehicle.get_speed(rl_id)
            headway = self._safe_headway(rl_id)

            agent_reward = 0.0
            agent_bonus = 0.0

            speed_error = abs(speed - self.target_speed)
            speed_reward = -speed_error / max(self.target_speed, 1e-3)
            agent_reward += 0.3 * speed_reward

            platoon_reward = 0.0
            if headway is not None:
                if 10.0 <= headway <= 30.0:
                    platoon_reward = 1.0
                    agent_bonus = 1.0
                elif headway > 30.0:
                    platoon_reward = max(-0.5, -0.01 * (headway - 30.0))
                else:
                    platoon_reward = 0.0
            agent_reward += 0.3 * platoon_reward

            if rl_id in self.prev_speeds:
                prev_speed = self.prev_speeds[rl_id]
                speed_change = abs(speed - prev_speed)
                if speed_change > 2.0:
                    agent_reward += 0.2 * (-speed_change / 10.0)
            self.prev_speeds[rl_id] = speed

            if headway is not None:
                if headway < 5.0:
                    agent_reward += 0.1 * (-1.0 * min(1.0, (5.0 - headway) / 5.0))
                elif headway < 8.0:
                    agent_reward += 0.1 * (-1.0 * (8.0 - headway) / 3.0)

            agent_reward += agent_bonus
            rewards[rl_id] = agent_reward

        return rewards

