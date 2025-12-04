"""Custom Flow environment for platooning with continuous control."""

from flow.envs.base import Env
import gym
import numpy as np


class PlatoonEnv(Env):
    """Single-agent environment that controls all RL vehicles jointly."""

    def __init__(self, env_params, sim_params, network, simulator="traci"):
        super().__init__(env_params, sim_params, network, simulator)
        self.prev_speeds = {}
        self.num_features = 5  # speed, headway, relative_speed, lane, normalized_speed
        self.target_speed = self.net_params.additional_params.get("speed_limit", 30.0)
        self.max_rl = self.initial_vehicles.num_rl_vehicles

    def reset(self):
        """Clear per-episode buffers and delegate to the base reset."""
        self.prev_speeds.clear()
        return super().reset()

    @property
    def action_space(self):
        return gym.spaces.Box(low=-3.0, high=3.0, shape=(self.max_rl,), dtype=np.float32)

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_rl * self.num_features,),
            dtype=np.float32,
        )

    def _safe_headway(self, veh_id):
        """Compute headway with modular arithmetic for circular roads.
        
        This fixes the wraparound issue where positions jump from 999.8m to 0.2m,
        which would give negative headway. Uses modular distance calculation.
        """
        headway = self.k.vehicle.get_headway(veh_id)
        if headway is None or headway < 0:  # no leader or invalid reading
            return 999.0
        
        # Get road length for modular calculation (if using circular/ring road)
        road_length = self.net_params.additional_params.get("length", 1000.0)
        
        # Check if this is a circular road (ring network)
        # For now, assume HighwayNetwork is straight, but handle wraparound if positions wrap
        # If headway seems too large (likely wraparound artifact), compute modular distance
        if headway > road_length * 0.5:
            # Likely wraparound: compute modular distance manually
            try:
                pos_follower = self.k.vehicle.get_position(veh_id)
                leader = self.k.vehicle.get_leader(veh_id)
                if leader:
                    pos_leader = self.k.vehicle.get_position(leader)
                    # Modular distance: (pos_leader - pos_follower) % road_length
                    # But handle negative case
                    delta = pos_leader - pos_follower
                    if delta < 0:
                        delta += road_length
                    return delta
            except:
                pass
        
        return headway

    def get_state(self):
        """Return concatenated observations for every RL vehicle."""
        rl_ids = sorted(self.k.vehicle.get_rl_ids())
        states = []
        for i in range(self.max_rl):
            if i < len(rl_ids):
                rl_id = rl_ids[i]
                speed = self.k.vehicle.get_speed(rl_id)
                headway = self._safe_headway(rl_id)
                leader = self.k.vehicle.get_leader(rl_id)
                if leader:
                    leader_speed = self.k.vehicle.get_speed(leader)
                    relative_speed = speed - leader_speed
                else:
                    relative_speed = 0.0
                lane = self.k.vehicle.get_lane(rl_id)
                normalized_speed = speed / max(self.target_speed, 1e-3)
                states.extend([speed, headway, relative_speed, lane, normalized_speed])
            else:
                states.extend([0.0] * self.num_features)
        return np.array(states, dtype=np.float32)

    def compute_reward(self, rl_actions, **kwargs):
        """Reward cooperative platooning behaviour for all RL vehicles."""
        rl_ids = self.k.vehicle.get_rl_ids()
        if len(rl_ids) == 0:
            return 0.0

        total_reward = 0.0
        positive_bonus = 0.0

        for rl_id in rl_ids:
            speed = self.k.vehicle.get_speed(rl_id)
            headway = self._safe_headway(rl_id)

            speed_error = abs(speed - self.target_speed)
            speed_reward = -speed_error / max(self.target_speed, 1e-3)
            total_reward += 0.3 * speed_reward

            if 10.0 <= headway <= 30.0:
                platoon_reward = 1.0
                positive_bonus += 1.0
            elif headway > 30.0:
                platoon_reward = -0.5 * (headway - 30.0) / 100.0
            else:
                platoon_reward = 0.0
            total_reward += 0.3 * platoon_reward

            if rl_id in self.prev_speeds:
                prev_speed = self.prev_speeds[rl_id]
                speed_change = abs(speed - prev_speed)
                if speed_change > 2.0:
                    total_reward += 0.2 * (-speed_change / 10.0)
            self.prev_speeds[rl_id] = speed

            if headway < 5.0:
                total_reward += 0.1 * (-5.0 * (5.0 - headway))
            elif headway < 8.0:
                total_reward += 0.1 * (-1.0 * (8.0 - headway))

        avg_reward = total_reward / len(rl_ids)
        bonus = positive_bonus / max(len(rl_ids), 1)
        return avg_reward + bonus

    def _apply_rl_actions(self, rl_actions):
        """Apply continuous acceleration commands to RL vehicles.
        
        Handles the case where there are more RL vehicles than actions
        (e.g., when new RL vehicles spawn via inflow). Extra vehicles
        get a default action of 0 (maintain speed).
        """
        if rl_actions is None:
            return

        rl_ids = sorted(self.k.vehicle.get_rl_ids())
        if len(rl_ids) == 0:
            return

        clipped = np.clip(rl_actions, -3.0, 3.0)
        
        # Handle case where there are more RL vehicles than actions
        # (can happen when new RL vehicles spawn via inflow)
        if len(clipped) < len(rl_ids):
            # Pad with zeros (maintain speed) for extra vehicles
            padded = np.zeros(len(rl_ids), dtype=np.float32)
            padded[:len(clipped)] = clipped
            clipped = padded
        elif len(clipped) > len(rl_ids):
            # Truncate if somehow we have more actions than vehicles
            clipped = clipped[:len(rl_ids)]
        
        self.k.vehicle.apply_acceleration(rl_ids, clipped)

    def additional_command(self):
        """Highlight RL vehicles in red for easier visual debugging."""
        for rl_id in self.k.vehicle.get_rl_ids():
            self.k.kernel_api.vehicle.setColor(rl_id, (255, 0, 0, 255))