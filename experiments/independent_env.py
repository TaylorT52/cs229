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
        """Stage-1 reward: simple longitudinal control without lane changing.

        Focus (throughput-oriented):
        - Strongly reward sustained high speed (throughput)
        - Mild, per-step cost for being below ~70% of target speed
        - Weaker penalties for small headways (safety still enforced, but softer)
        - Very small smoothness penalty on large accelerations

        This is intentionally simpler and less harsh than the full platooning
        reward used in the single-agent environment.
        """
        rl_ids = self.k.vehicle.get_rl_ids()
        if len(rl_ids) == 0:
            return {}

        rewards = {}

        for rl_id in rl_ids:
            speed = self.k.vehicle.get_speed(rl_id)
            headway = self._safe_headway(rl_id)

            agent_reward = 0.0

            # 1) Reward moving near target speed (normalized to [0, 1])
            speed_norm = speed / max(self.target_speed, 1e-3)
            speed_norm_clipped = max(0.0, min(speed_norm, 1.0))
            # Prefer high sustained speeds (throughput)
            agent_reward += 2.0 * speed_norm_clipped

            # 2) Penalty for being very slow / near-stop (keep but soften)
            if speed < 0.5:
                # Strong penalty for essentially stopped vehicles
                agent_reward -= 3.0
            elif speed < 0.3 * self.target_speed:
                # Linearly increasing penalty as speed drops below 30% of target
                slow_frac = 1.0 - speed / max(0.3 * self.target_speed, 1e-3)
                agent_reward -= 1.0 * slow_frac

            # 2b) Mild per-step cost for being below ~70% of target speed
            if speed < 0.7 * self.target_speed:
                deficit = 0.7 * self.target_speed - speed
                deficit_frac = deficit / max(0.7 * self.target_speed, 1e-3)
                agent_reward -= 0.5 * deficit_frac

            # 3) Safety: penalize very small headways only (keep simple)
            if headway is not None:
                if headway < 3.0:
                    # Extremely close: strong penalty (but weaker than before)
                    close_frac = (3.0 - headway) / 3.0
                    agent_reward -= 1.0 * min(close_frac, 1.0)
                elif headway < 6.0:
                    # Moderately too close: mild penalty
                    close_frac = (6.0 - headway) / 3.0
                    agent_reward -= 0.2 * min(close_frac, 1.0)

            # 4) Smoothness: very small penalty on large accelerations
            if rl_id in self.prev_speeds:
                prev_speed = self.prev_speeds[rl_id]
                accel = speed - prev_speed
                if abs(accel) > 2.0:
                    jerk_frac = (abs(accel) - 2.0) / 3.0
                    agent_reward -= 0.1 * min(jerk_frac, 1.0)
            self.prev_speeds[rl_id] = speed

            rewards[rl_id] = agent_reward

        return rewards

