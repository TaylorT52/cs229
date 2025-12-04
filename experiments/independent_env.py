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
        rl_speeds = [self.k.vehicle.get_speed(rid) for rid in rl_ids]
        avg_rl_speed = (
            sum(rl_speeds) / len(rl_speeds) if len(rl_speeds) > 0 else 0.0
        )

        for rl_id in rl_ids:
            speed = self.k.vehicle.get_speed(rl_id)
            headway = self._safe_headway(rl_id)
            leader = self.k.vehicle.get_leader(rl_id)
            leader_speed = self.k.vehicle.get_speed(leader) if leader else None
            is_slow_leader = (
                leader_speed is not None and leader_speed < 0.7 * self.target_speed
            )

            agent_reward = 0.0

            # 1) Reward moving near target speed (normalized to [0, 1])
            speed_norm = speed / max(self.target_speed, 1e-3)
            speed_norm_clipped = max(0.0, min(speed_norm, 1.0))
            # Prefer high sustained speeds (throughput)
            agent_reward += 2.5 * speed_norm_clipped

            # Bonus for maintaining near-target speeds (>= 90% of target)
            if speed >= 0.9 * self.target_speed:
                fast_frac = (speed - 0.9 * self.target_speed) / max(
                    0.1 * self.target_speed, 1e-3
                )
                agent_reward += 0.5 * min(max(fast_frac, 0.0), 1.0)

            # 2) Penalty for being very slow / near-stop (keep but soften)
            if speed < 0.5:
                # Strong penalty for essentially stopped vehicles
                agent_reward -= 3.0
            elif speed < 0.3 * self.target_speed:
                # Linearly increasing penalty as speed drops below 30% of target
                slow_frac = 1.0 - speed / max(0.3 * self.target_speed, 1e-3)
                agent_reward -= 1.0 * slow_frac

            # 2b) Mild per-step cost for being below ~70% of target speed,
            # but only when headway is large (i.e., free to accelerate).
            if (
                headway is not None
                and headway > 20.0
                and speed < 0.7 * self.target_speed
            ):
                deficit = 0.7 * self.target_speed - speed
                deficit_frac = deficit / max(0.7 * self.target_speed, 1e-3)
                agent_reward -= 0.8 * deficit_frac

            # Extra bonus when effectively free (no close or slow leader)
            # Approximate "free" as having very large headway.
            if headway is None or headway > 40.0:
                free_speed_bonus = speed / max(self.target_speed, 1e-3)
                agent_reward += 1.0 * min(max(free_speed_bonus, 0.0), 1.0)

            # Penalty for lingering behind a slow leader in close proximity
            if is_slow_leader and headway is not None and headway < 20.0:
                stuck_frac = (20.0 - headway) / 20.0
                agent_reward -= 0.5 * min(stuck_frac, 1.0)

            # Synchronization reward: encourage matching other RL speeds
            if avg_rl_speed > 0:
                sync_term = 1.0 - min(
                    abs(speed - avg_rl_speed) / max(self.target_speed, 1e-3), 1.0
                )
                agent_reward += 0.5 * sync_term

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
                    agent_reward -= 0.05 * min(jerk_frac, 1.0)
            self.prev_speeds[rl_id] = speed

            # Reward lane changes that actually improve speed; penalize the opposite.
            if rl_id in self.last_lane_change and rl_id in self.prev_speeds_before_lc:
                time_since_lc = self.time_counter - self.last_lane_change[rl_id]
                if time_since_lc < 5.0:
                    prev_speed_before_lc = self.prev_speeds_before_lc[rl_id]
                    speed_gain = speed - prev_speed_before_lc
                    if speed_gain > 0.5:
                        agent_reward += 0.6 * min(speed_gain / 5.0, 1.0)
                    elif speed_gain < -0.2:
                        agent_reward -= 0.4 * min(abs(speed_gain) / 3.0, 1.0)
                # Consume the stored value so it only contributes once
                del self.prev_speeds_before_lc[rl_id]

            rewards[rl_id] = agent_reward

        return rewards

