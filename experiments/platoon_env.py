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
        # Lane change parameters
        self.lane_change_enabled = env_params.additional_params.get("lane_change_enabled", False)
        self.lane_change_duration = env_params.additional_params.get("lane_change_duration", 5.0)  # seconds
        self.last_lane_change = {}  # Track last lane change time for each vehicle

    def reset(self):
        """Clear per-episode buffers and delegate to the base reset."""
        self.prev_speeds.clear()
        self.last_lane_change.clear()
        return super().reset()

    @property
    def action_space(self):
        if self.lane_change_enabled:
            # Actions: [accel_0, lane_change_0, accel_1, lane_change_1, ...]
            # Lane change: -1 (left), 0 (stay), 1 (right)
            return gym.spaces.Box(
                low=np.array([-3.0, -1.0] * self.max_rl, dtype=np.float32),
                high=np.array([3.0, 1.0] * self.max_rl, dtype=np.float32),
                dtype=np.float32
            )
        else:
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
        """Apply continuous acceleration and optional lane change commands to RL vehicles.
        
        Handles the case where there are more RL vehicles than actions
        (e.g., when new RL vehicles spawn via inflow). Extra vehicles
        get a default action of 0 (maintain speed, no lane change).
        """
        if rl_actions is None:
            return

        rl_ids = sorted(self.k.vehicle.get_rl_ids())
        if len(rl_ids) == 0:
            return

        if self.lane_change_enabled:
            # Actions are interleaved: [accel_0, lane_change_0, accel_1, lane_change_1, ...]
            accelerations = rl_actions[::2]  # Every even index
            lane_changes = rl_actions[1::2]  # Every odd index
            
            # Clip accelerations
            accelerations = np.clip(accelerations, -3.0, 3.0)
            # Discretize lane changes: continuous -> discrete (-1, 0, 1)
            # Threshold at 0.3 to avoid noise
            lane_changes_discrete = np.zeros_like(lane_changes, dtype=np.int32)
            lane_changes_discrete[lane_changes > 0.3] = 1   # Right
            lane_changes_discrete[lane_changes < -0.3] = -1  # Left
            # Otherwise stays 0 (no lane change)
            lane_changes = lane_changes_discrete.astype(np.float32)
            
            # Handle case where there are more RL vehicles than actions
            num_actions = len(accelerations)
            if num_actions < len(rl_ids):
                # Pad with zeros for extra vehicles
                accel_padded = np.zeros(len(rl_ids), dtype=np.float32)
                accel_padded[:num_actions] = accelerations
                accelerations = accel_padded
                
                lc_padded = np.zeros(len(rl_ids), dtype=np.float32)
                lc_padded[:num_actions] = lane_changes
                lane_changes = lc_padded
            elif num_actions > len(rl_ids):
                # Truncate if somehow we have more actions than vehicles
                accelerations = accelerations[:len(rl_ids)]
                lane_changes = lane_changes[:len(rl_ids)]
            
            # Apply lane change cooldown - prevent too frequent lane changes
            current_time = self.time_counter
            for i, veh_id in enumerate(rl_ids):
                last_lc_time = self.last_lane_change.get(veh_id, -float('inf'))
                time_since_lc = current_time - last_lc_time
                
                # If vehicle changed lanes recently, prevent new lane change
                if time_since_lc < self.lane_change_duration:
                    lane_changes[i] = 0.0
                # If lane change is being performed, record it
                elif lane_changes[i] != 0:  # Non-zero means lane change attempted
                    # We'll update last_lane_change after checking if lane changed
                    pass
            
            # Apply actions
            self.k.vehicle.apply_acceleration(rl_ids, accelerations)
            self.k.vehicle.apply_lane_change(rl_ids, direction=lane_changes)
            
            # Update last lane change times for vehicles that attempted lane changes
            for i, veh_id in enumerate(rl_ids):
                if lane_changes[i] != 0:
                    # Record the attempt (SUMO will handle the actual lane change)
                    self.last_lane_change[veh_id] = current_time
        else:
            # No lane changes - just acceleration
            clipped = np.clip(rl_actions, -3.0, 3.0)
            
            # Handle case where there are more RL vehicles than actions
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