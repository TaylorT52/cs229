import numpy as np
from platoon_env import PlatoonEnv


class IndependentPlatoonEnv(PlatoonEnv):
    def __init__(self, env_params, sim_params, network, simulator="traci"):
        super().__init__(env_params, sim_params, network, simulator)
        self.prev_lanes = {}

    def reset(self):
        self.prev_lanes.clear()
        return super().reset()

    def compute_reward(self, rl_actions, **kwargs):
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

            agent_reward = 0.1
            speed_ratio = speed / max(self.target_speed, 1e-3)
            if speed_ratio > 1.0:
                speed_reward = -0.5 * (speed_ratio - 1.0)
            else:
                speed_reward = speed_ratio - 1.0 
            agent_reward += 0.3 * speed_reward
            if headway is not None:
                if 10.0 <= headway <= 30.0:
                    platoon_reward = 1.0
                elif 30.0 < headway <= 60.0:
                    platoon_reward = 0.5 - 0.01 * (headway - 30.0)
                elif headway > 60.0:
                    platoon_reward = max(-0.3, 0.2 - 0.01 * (headway - 60.0))
                elif 5.0 <= headway < 10.0:
                    platoon_reward = 0.5 * (headway - 5.0) / 5.0
                else:
                    platoon_reward = 0.0
            else:
                platoon_reward = 0.0
            agent_reward += 0.3 * platoon_reward
            if rl_id in self.prev_speeds:
                prev_speed = self.prev_speeds[rl_id]
                speed_change = abs(speed - prev_speed)
                stability_penalty = -min(0.5, speed_change / 5.0)
                agent_reward += 0.2 * stability_penalty
            self.prev_speeds[rl_id] = speed

            if headway is not None and headway < 5.0:
                safety_penalty = -min(1.0, (5.0 - headway) / 5.0)
                agent_reward += 0.15 * safety_penalty
            elif headway is not None and headway < 8.0:
                safety_penalty = -0.2 * (8.0 - headway) / 3.0
                agent_reward += 0.15 * safety_penalty
            current_lane = self.k.vehicle.get_lane(rl_id)
            if rl_id in self.prev_lanes:
                prev_lane = self.prev_lanes[rl_id]
                if current_lane != prev_lane:
                    agent_reward += -0.3
            self.prev_lanes[rl_id] = current_lane

            rewards[rl_id] = agent_reward

        return rewards

