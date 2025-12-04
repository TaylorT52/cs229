"""Custom Flow environment for platooning with continuous control."""

from flow.envs.base import Env
import gym
import numpy as np


class PlatoonEnv(Env):
    """Single-agent environment that controls all RL vehicles jointly."""

    def __init__(self, env_params, sim_params, network, simulator="traci"):
        super().__init__(env_params, sim_params, network, simulator)
        self.prev_speeds = {}
        self.prev_lanes = {}  # Track previous lane for lateral velocity
        self.target_speed = self.net_params.additional_params.get("speed_limit", 30.0)
        self.max_rl = self.initial_vehicles.num_rl_vehicles
        # Lane change parameters
        self.lane_change_enabled = env_params.additional_params.get("lane_change_enabled", False)
        self.lane_change_duration = env_params.additional_params.get("lane_change_duration", 3.0)  # seconds (increased to reduce tweakiness)
        self.last_lane_change = {}  # Track last lane change time for each vehicle
        self.last_lane_change_direction = {}  # Track last direction to prevent oscillation
        # Smoothing for lane change actions (exponential moving average)
        self.lane_change_smoothing = 0.6  # Balanced: responsive but stable
        self.smoothed_lane_changes = {}  # Track smoothed lane change values
        self.prev_speeds_before_lc = {}  # Track speed before lane change to reward improvements
        
        # Number of features per agent
        # Base features (always used):
        #   speed, headway, relative_speed, lane, normalized_speed,
        #   accel (delta speed), ttc, leader_speed_norm, is_slow_leader
        # Lane-change features (only when lane_change_enabled=True):
        #   left_lane_free, dist_front_left, dist_rear_left, left_rel_speed,
        #   right_lane_free, dist_front_right, dist_rear_right, right_rel_speed, lateral_velocity
        if self.lane_change_enabled:
            # 9 base longitudinal features + 9 lane-related = 18 total
            self.num_features = 18
        else:
            # Rich longitudinal state, no lateral information
            self.num_features = 9

    def reset(self):
        """Clear per-episode buffers and delegate to the base reset."""
        self.prev_speeds.clear()
        self.prev_lanes.clear()
        self.last_lane_change.clear()
        self.last_lane_change_direction.clear()
        self.smoothed_lane_changes.clear()
        self.prev_speeds_before_lc.clear()
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
        """Return concatenated observations for every RL vehicle.
        
        If lane_change_enabled, includes lane-related features:
        - Base: speed, headway, relative_speed, lane, normalized_speed
        - Lane: left_lane_free, dist_front_left, dist_rear_left,
                right_lane_free, dist_front_right, dist_rear_right, lateral_velocity
        """
        rl_ids = sorted(self.k.vehicle.get_rl_ids())
        states = []
        max_lanes = self.net_params.additional_params.get("lanes", 4)
        
        for i in range(self.max_rl):
            if i < len(rl_ids):
                rl_id = rl_ids[i]
                
                # Base features
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
                # Acceleration estimate from previous step
                prev_speed = self.prev_speeds.get(rl_id, speed)
                accel = speed - prev_speed
                # Time-to-collision (TTC) estimate
                if leader and relative_speed > 0.0 and headway is not None:
                    ttc = headway / max(relative_speed, 1e-3)
                    ttc = min(ttc, 10.0) / 10.0  # normalize to [0, 1]
                else:
                    ttc = 1.0  # safe by default
                # Normalized leader speed and "slow leader" flag
                if leader:
                    leader_speed = self.k.vehicle.get_speed(leader)
                    leader_speed_norm = leader_speed / max(self.target_speed, 1e-3)
                    is_slow_leader = 1.0 if leader_speed < 0.7 * self.target_speed else 0.0
                else:
                    leader_speed_norm = 1.0
                    is_slow_leader = 0.0

                state_features = [
                    speed,
                    headway,
                    relative_speed,
                    lane,
                    normalized_speed,
                    accel,
                    ttc,
                    leader_speed_norm,
                    is_slow_leader,
                ]

                # Lane-related features (only if lane changes enabled)
                if self.lane_change_enabled:
                    # Get lane leaders and followers for all lanes
                    lane_leaders = self.k.vehicle.get_lane_leaders(rl_id)
                    lane_followers = self.k.vehicle.get_lane_followers(rl_id)
                    lane_headways = self.k.vehicle.get_lane_headways(rl_id)
                    lane_tailways = self.k.vehicle.get_lane_tailways(rl_id)
                    
                    # Ensure we have data for all lanes (pad if needed)
                    while len(lane_leaders) < max_lanes:
                        lane_leaders.append('')
                        lane_followers.append('')
                        lane_headways.append(999.0)
                        lane_tailways.append(999.0)
                    
                    # Left lane (higher index = left)
                    left_lane_idx = lane + 1
                    if left_lane_idx < max_lanes:
                        left_leader = lane_leaders[left_lane_idx] if left_lane_idx < len(lane_leaders) else ''
                        left_follower = lane_followers[left_lane_idx] if left_lane_idx < len(lane_followers) else ''
                        left_lane_free = 1.0 if (left_leader == '' and left_follower == '') else 0.0
                        dist_front_left = lane_headways[left_lane_idx] if left_lane_idx < len(lane_headways) else 999.0
                        dist_rear_left = lane_tailways[left_lane_idx] if left_lane_idx < len(lane_tailways) else 999.0
                    else:
                        left_lane_free = 0.0  # No left lane available
                        dist_front_left = 999.0
                        dist_rear_left = 999.0
                    
                    # Right lane (lower index = right)
                    right_lane_idx = lane - 1
                    if right_lane_idx >= 0:
                        right_leader = lane_leaders[right_lane_idx] if right_lane_idx < len(lane_leaders) else ''
                        right_follower = lane_followers[right_lane_idx] if right_lane_idx < len(lane_followers) else ''
                        right_lane_free = 1.0 if (right_leader == '' and right_follower == '') else 0.0
                        dist_front_right = lane_headways[right_lane_idx] if right_lane_idx < len(lane_headways) else 999.0
                        dist_rear_right = lane_tailways[right_lane_idx] if right_lane_idx < len(lane_tailways) else 999.0
                    else:
                        right_lane_free = 0.0  # No right lane available
                        dist_front_right = 999.0
                        dist_rear_right = 999.0
                    
                    # Lateral velocity (change in lane position)
                    if rl_id in self.prev_lanes:
                        lateral_velocity = lane - self.prev_lanes[rl_id]
                    else:
                        lateral_velocity = 0.0
                    self.prev_lanes[rl_id] = lane
                    
                    # Normalize distances (cap at reasonable max)
                    dist_front_left = min(dist_front_left, 200.0) / 200.0
                    dist_rear_left = min(dist_rear_left, 200.0) / 200.0
                    dist_front_right = min(dist_front_right, 200.0) / 200.0
                    dist_rear_right = min(dist_rear_right, 200.0) / 200.0
                    
                    # Add relative speeds in adjacent lanes (CRITICAL for lane change decisions)
                    left_rel_speed = 0.0
                    right_rel_speed = 0.0
                    
                    if left_lane_idx < max_lanes and left_lane_idx < len(lane_leaders):
                        if left_leader != '':
                            left_leader_speed = self.k.vehicle.get_speed(left_leader)
                            left_rel_speed = (speed - left_leader_speed) / max(self.target_speed, 1e-3)  # Normalized
                        else:
                            left_rel_speed = 1.0  # No leader = can accelerate freely
                    
                    if right_lane_idx >= 0 and right_lane_idx < len(lane_leaders):
                        if right_leader != '':
                            right_leader_speed = self.k.vehicle.get_speed(right_leader)
                            right_rel_speed = (speed - right_leader_speed) / max(self.target_speed, 1e-3)  # Normalized
                        else:
                            right_rel_speed = 1.0  # No leader = can accelerate freely
                    
                    state_features.extend([
                        left_lane_free,
                        dist_front_left,
                        dist_rear_left,
                        left_rel_speed,  # NEW: relative speed in left lane
                        right_lane_free,
                        dist_front_right,
                        dist_rear_right,
                        right_rel_speed,  # NEW: relative speed in right lane
                        lateral_velocity
                    ])
                # Update prev_speeds for next step (used by both obs and reward)
                self.prev_speeds[rl_id] = speed

                states.extend(state_features)
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
        lane_change_penalty = 0.0

        # Track lane distribution for reward
        lane_counts = {}
        for rl_id in rl_ids:
            try:
                lane = self.k.vehicle.get_lane(rl_id)
                lane_counts[lane] = lane_counts.get(lane, 0) + 1
            except:
                pass

        for rl_id in rl_ids:
            speed = self.k.vehicle.get_speed(rl_id)
            headway = self._safe_headway(rl_id)

            speed_error = abs(speed - self.target_speed)
            speed_reward = -speed_error / max(self.target_speed, 1e-3)
            # Increased weight for speed to prevent speed collapse
            total_reward += 0.5 * speed_reward

            # Platoon reward (cooperative following) - but safety takes priority
            if headway is not None:
                if 10.0 <= headway <= 30.0:
                    platoon_reward = 1.0
                    positive_bonus += 1.0
                elif headway > 30.0:
                    # Cap the penalty to prevent extreme negative rewards
                    # Max penalty is -0.5 even for very large headways
                    platoon_reward = max(-0.5, -0.01 * (headway - 30.0))
                else:
                    # Headway < 10.0 - safety penalties already applied above
                    # Small additional penalty for being too close for platooning
                    platoon_reward = -0.2 * (10.0 - headway) / 10.0
            else:
                # No leader - small penalty instead of large negative reward
                platoon_reward = -0.1
            # Increased weight for platooning to encourage coordination
            total_reward += 0.4 * platoon_reward

            if rl_id in self.prev_speeds:
                prev_speed = self.prev_speeds[rl_id]
                speed_change = abs(speed - prev_speed)
                if speed_change > 2.0:
                    total_reward += 0.2 * (-speed_change / 10.0)
                
                # Reward for accelerating when safe (helps avoid slow-speed local minima)
                if headway is not None and headway > 15.0:
                    acceleration = speed - prev_speed
                    if acceleration > 0:  # Accelerating
                        total_reward += 0.05 * min(acceleration / 2.0, 1.0)  # Small reward for acceleration
            
            self.prev_speeds[rl_id] = speed

            # SCALED-DOWN SAFETY PENALTIES (70-80% reduction to prevent over-constraint)
            if headway is not None:
                if headway < 3.0:  # Extremely dangerous (< 3m)
                    safety_penalty = -0.8 * (3.0 - headway) / 3.0  # Reduced from -2.0
                    total_reward += safety_penalty
                elif headway < 5.0:  # Dangerous (< 5m)
                    safety_penalty = -0.4 * (5.0 - headway) / 2.0  # Reduced from -1.0
                    total_reward += safety_penalty
                elif headway < 8.0:  # Too close (< 8m)
                    safety_penalty = -0.2 * (8.0 - headway) / 3.0  # Reduced from -0.5
                    total_reward += safety_penalty
                elif headway < 10.0:  # Below optimal but acceptable
                    safety_penalty = -0.05 * (10.0 - headway) / 2.0  # Reduced from -0.1
                    total_reward += safety_penalty
                elif 10.0 <= headway <= 30.0:  # Optimal safe range
                    # Reward for maintaining safe headway (already in platoon_reward, but reinforce)
                    safety_reward = 0.2 * (1.0 - abs(headway - 20.0) / 20.0)  # Peak at 20m
                    total_reward += safety_reward
                
                # Time-to-collision (TTC) based penalty - SCALED DOWN
                if headway < 20.0:  # Only check TTC when close
                    leader = self.k.vehicle.get_leader(rl_id)
                    if leader:
                        leader_speed = self.k.vehicle.get_speed(leader)
                        relative_speed = speed - leader_speed
                        if relative_speed > 0:  # Approaching leader
                            ttc = headway / relative_speed
                            if ttc < 2.0:  # TTC < 2 seconds is dangerous
                                ttc_penalty = -0.6 * (2.0 - ttc) / 2.0  # Reduced from -1.5
                                total_reward += ttc_penalty
                            elif ttc < 5.0:  # TTC < 5 seconds is concerning
                                ttc_penalty = -0.1 * (5.0 - ttc) / 3.0  # Reduced from -0.3
                                total_reward += ttc_penalty
            
            # Penalize excessive lane changes (encourage stable lane usage) - REDUCED PENALTY
            if self.lane_change_enabled and rl_id in self.last_lane_change:
                time_since_lc = self.time_counter - self.last_lane_change[rl_id]
                # If changed lanes very recently (< 3 seconds), small penalty
                if time_since_lc < 3.0:
                    lane_change_penalty += 0.025 * (3.0 - time_since_lc) / 3.0  # Reduced from 0.05
            
            # Reward lane distribution and penalize staying behind slow traffic
            if self.lane_change_enabled:
                try:
                    lane = self.k.vehicle.get_lane(rl_id)
                    # Reward being in a less crowded lane
                    vehicles_in_lane = lane_counts.get(lane, 0)
                    if vehicles_in_lane > 1:
                        # Penalty for being in crowded lane
                        lane_distribution_reward = -0.1 * (vehicles_in_lane - 1) / max(len(rl_ids), 1)
                        total_reward += lane_distribution_reward
                    
                    # EXPLICIT lane-change rewards (makes signal stable)
                    # Track if lane change just occurred
                    if rl_id in self.last_lane_change:
                        time_since_lc = self.time_counter - self.last_lane_change[rl_id]
                        if time_since_lc < 0.5:  # Just changed lanes (< 0.5s ago)
                            # Check if lane change was beneficial
                            leader = self.k.vehicle.get_leader(rl_id)
                            if leader:
                                leader_speed = self.k.vehicle.get_speed(leader)
                                if leader_speed > self.target_speed * 0.8:  # Good: leader is fast
                                    total_reward += 0.5  # Reward successful lane change
                                elif leader_speed < self.target_speed * 0.5:  # Bad: ended up behind slower car
                                    total_reward += -0.5  # Penalize bad lane change
                            
                            # Check speed improvement from lane change
                            if rl_id in self.prev_speeds_before_lc:
                                prev_speed_before_lc = self.prev_speeds_before_lc[rl_id]
                                speed_improvement = speed - prev_speed_before_lc
                                if speed_improvement > 0.5:  # Significant speed improvement
                                    total_reward += 0.6 * min(speed_improvement / 5.0, 1.0)  # Reward improvement
                                elif speed <= prev_speed_before_lc:  # Speed didn't improve
                                    total_reward += -0.3  # Small penalty for unnecessary change
                                # Clear the tracking after checking
                                del self.prev_speeds_before_lc[rl_id]
                    
                    # DETECT SLOW LEADER AND REWARD/PENALIZE LANE CHANGES ACCORDINGLY
                    leader = self.k.vehicle.get_leader(rl_id)
                    if leader:
                        leader_speed = self.k.vehicle.get_speed(leader)
                        # Check if leader is significantly slower than target speed
                        slow_leader_threshold = self.target_speed * 0.7  # Leader < 70% of target speed
                        is_slow_leader = leader_speed < slow_leader_threshold
                        
                        if is_slow_leader:
                            # Get lane information to check if lane change is possible
                            max_lanes = self.net_params.additional_params.get("lanes", 4)
                            lane_leaders = self.k.vehicle.get_lane_leaders(rl_id)
                            lane_followers = self.k.vehicle.get_lane_followers(rl_id)
                            lane_headways = self.k.vehicle.get_lane_headways(rl_id)
                            
                            # Check adjacent lanes for better options
                            left_lane_idx = lane + 1
                            right_lane_idx = lane - 1
                            
                            better_lane_available = False
                            if left_lane_idx < max_lanes and left_lane_idx < len(lane_leaders):
                                left_leader = lane_leaders[left_lane_idx] if left_lane_idx < len(lane_leaders) else ''
                                left_follower = lane_followers[left_lane_idx] if left_lane_idx < len(lane_followers) else ''
                                # Check if left lane is free or has faster traffic
                                if left_leader == '' and left_follower == '':
                                    better_lane_available = True
                                elif left_leader != '':
                                    left_leader_speed = self.k.vehicle.get_speed(left_leader)
                                    if left_leader_speed > leader_speed + 1.0:  # At least 1 m/s faster
                                        better_lane_available = True
                            
                            if not better_lane_available and right_lane_idx >= 0 and right_lane_idx < len(lane_leaders):
                                right_leader = lane_leaders[right_lane_idx] if right_lane_idx < len(lane_leaders) else ''
                                right_follower = lane_followers[right_lane_idx] if right_lane_idx < len(lane_followers) else ''
                                # Check if right lane is free or has faster traffic
                                if right_leader == '' and right_follower == '':
                                    better_lane_available = True
                                elif right_leader != '':
                                    right_leader_speed = self.k.vehicle.get_speed(right_leader)
                                    if right_leader_speed > leader_speed + 1.0:  # At least 1 m/s faster
                                        better_lane_available = True
                            
                            # Penalize staying behind slow leader when better lane is available - REDUCED PENALTY
                            if better_lane_available:
                                speed_difference = slow_leader_threshold - leader_speed
                                # Reduced penalty to encourage lane changes
                                stuck_penalty = -0.25 * (speed_difference / max(slow_leader_threshold, 1e-3))  # Reduced from -0.5
                                total_reward += stuck_penalty
                                
                                # Additional penalty if speed is also low (really stuck)
                                if speed < self.target_speed * 0.8:
                                    total_reward += -0.3 * (1.0 - speed / (self.target_speed * 0.8))
                            
                            # Reward if vehicle recently changed lanes to avoid slow traffic - INCREASED REWARD
                            if rl_id in self.last_lane_change:
                                time_since_lc = self.time_counter - self.last_lane_change[rl_id]
                                # If changed lanes recently (< 5 seconds) and speed improved
                                if time_since_lc < 5.0 and speed > leader_speed + 0.5:
                                    lane_change_reward = 0.8 * min((speed - leader_speed) / 5.0, 1.0)  # Increased from 0.4
                                    total_reward += lane_change_reward
                except:
                    pass
            
            # Additional reward for maintaining target speed (encourage RL cars not to slow down traffic)
            speed_maintenance_reward = 0.0
            if speed >= self.target_speed * 0.9:  # Within 90% of target speed
                speed_maintenance_reward = 0.2 * (speed / max(self.target_speed, 1e-3))
            elif speed < self.target_speed * 0.7:  # Significantly below target
                speed_maintenance_reward = -0.3 * (1.0 - speed / max(self.target_speed * 0.7, 1e-3))
            total_reward += speed_maintenance_reward
            
            # Penalty for being TOO slow (prevents speed collapse)
            if speed < self.target_speed * 0.5:  # Below 50% of target speed
                slow_speed_penalty = -0.5  # Fixed penalty for being too slow
                total_reward += slow_speed_penalty
            
            # STRONG penalty for stopping or near-stopping (prevents stoppages) - INCREASED
            if speed < 0.5:  # Nearly stopped (< 0.5 m/s)
                stop_penalty = -3.0 * (1.0 - speed / 0.5)  # Increased from -1.0 to -3.0
                total_reward += stop_penalty
            
            # Penalty for rapid deceleration that could cause stoppages
            if rl_id in self.prev_speeds:
                prev_speed = self.prev_speeds[rl_id]
                deceleration = prev_speed - speed
                if deceleration > 3.0:  # Rapid deceleration (> 3 m/s per step)
                    total_reward += -0.5 * (deceleration - 3.0) / 3.0  # Penalty for excessive braking

        avg_reward = total_reward / len(rl_ids)
        bonus = positive_bonus / max(len(rl_ids), 1)
        lane_change_penalty_avg = lane_change_penalty / max(len(rl_ids), 1)
        return avg_reward + bonus - lane_change_penalty_avg

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
            
            # Handle case where there are more RL vehicles than actions
            # Pad BEFORE smoothing to avoid index errors
            num_actions = len(accelerations)
            num_vehicles = len(rl_ids)
            
            if num_actions < num_vehicles:
                # Pad with zeros for extra vehicles (newly spawned via inflow)
                accel_padded = np.zeros(num_vehicles, dtype=np.float32)
                accel_padded[:num_actions] = accelerations
                accelerations = accel_padded
                
                lc_padded = np.zeros(num_vehicles, dtype=np.float32)
                lc_padded[:num_actions] = lane_changes
                lane_changes = lc_padded
            elif num_actions > num_vehicles:
                # Truncate if somehow we have more actions than vehicles
                accelerations = accelerations[:num_vehicles]
                lane_changes = lane_changes[:num_vehicles]
            
            # Now smooth lane change actions to reduce oscillation
            # At this point, lane_changes has the same length as rl_ids
            smoothed_lc = np.zeros_like(lane_changes, dtype=np.float32)
            for i, veh_id in enumerate(rl_ids):
                if veh_id in self.smoothed_lane_changes:
                    # Exponential moving average
                    smoothed_lc[i] = (self.lane_change_smoothing * self.smoothed_lane_changes[veh_id] + 
                                     (1 - self.lane_change_smoothing) * lane_changes[i])
                else:
                    smoothed_lc[i] = lane_changes[i]
                self.smoothed_lane_changes[veh_id] = smoothed_lc[i]
            
            # Discretize lane changes with HYSTERESIS to prevent oscillation
            # Higher threshold to turn ON, lower to turn OFF (prevents tweakiness)
            lane_changes_discrete = np.zeros_like(smoothed_lc, dtype=np.int32)
            
            for i, veh_id in enumerate(rl_ids):
                current_smoothed = smoothed_lc[i]
                last_direction = self.last_lane_change_direction.get(veh_id, 0)
                
                # Hysteresis: different thresholds for turning on vs off
                if last_direction == 0:  # Currently no lane change
                    # Need stronger signal to start lane change
                    if current_smoothed > 0.2:  # Right
                        lane_changes_discrete[i] = 1
                    elif current_smoothed < -0.2:  # Left
                        lane_changes_discrete[i] = -1
                else:  # Currently changing lanes
                    # Need weaker signal to continue, but stronger opposite to reverse
                    if last_direction == 1:  # Was going right
                        if current_smoothed > 0.05:  # Continue right
                            lane_changes_discrete[i] = 1
                        elif current_smoothed < -0.25:  # Strong opposite to reverse
                            lane_changes_discrete[i] = -1
                        # Otherwise stays 0 (stop lane change)
                    elif last_direction == -1:  # Was going left
                        if current_smoothed < -0.05:  # Continue left
                            lane_changes_discrete[i] = -1
                        elif current_smoothed > 0.25:  # Strong opposite to reverse
                            lane_changes_discrete[i] = 1
                        # Otherwise stays 0 (stop lane change)
            
            lane_changes = lane_changes_discrete.astype(np.float32)
            
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
                    # Store speed before lane change to reward improvements
                    try:
                        self.prev_speeds_before_lc[veh_id] = self.k.vehicle.get_speed(veh_id)
                    except:
                        pass
                    # Less restrictive safety check - only prevent if dangerously close
                    # Allow lane changes even when close if it might help avoid slowdowns
                    try:
                        current_lane = self.k.vehicle.get_lane(veh_id)
                        leader = self.k.vehicle.get_leader(veh_id)
                        if leader:
                            headway = self.k.vehicle.get_headway(veh_id)
                            # Only prevent if extremely close (< 2.0m) - more permissive
                            if headway is not None and headway < 2.0:
                                lane_changes[i] = 0.0
                    except:
                        # If we can't check, allow the lane change attempt
                        pass
            
            # Apply actions
            self.k.vehicle.apply_acceleration(rl_ids, accelerations)
            self.k.vehicle.apply_lane_change(rl_ids, direction=lane_changes)
            
            # Update last lane change times and directions for vehicles that attempted lane changes
            for i, veh_id in enumerate(rl_ids):
                if lane_changes[i] != 0:
                    # Record the attempt (SUMO will handle the actual lane change)
                    self.last_lane_change[veh_id] = current_time
                    self.last_lane_change_direction[veh_id] = int(lane_changes[i])
                else:
                    # If no lane change, check if we've been in cooldown long enough to reset direction
                    if veh_id in self.last_lane_change:
                        time_since_lc = current_time - self.last_lane_change[veh_id]
                        if time_since_lc > self.lane_change_duration * 2:  # Reset after double cooldown
                            self.last_lane_change_direction[veh_id] = 0
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