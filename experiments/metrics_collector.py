import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional
import json
import os


class MetricsCollector:
    def __init__(self, policy_type: str = "unknown"):
        self.policy_type = policy_type

        self.step_data = []
        self.metrics = defaultdict(list)
        self.vehicle_history = defaultdict(list)
        self.collisions = []
        self.near_collisions = []
        self.action_history = defaultdict(list)
        self.disturbance_applied = False
        self.disturbance_time = None
        
    def collect_step(self, env, action_dict: Dict, reward_dict: Dict, step: int, time: float):
        k = env.env.k 
    
        rl_ids = sorted(k.vehicle.get_rl_ids())
        human_ids = sorted(k.vehicle.get_human_ids())
        all_ids = rl_ids + human_ids
        
        if len(rl_ids) == 0:
            return
        
        step_record = {
            "step": step,
            "time": time,
            "num_rl_vehicles": len(rl_ids),
            "num_human_vehicles": len(human_ids),
        }
        
        rl_data = {}
        for veh_id in rl_ids:
            try:
                speed = k.vehicle.get_speed(veh_id)
                position = k.vehicle.get_position(veh_id)
                lane = k.vehicle.get_lane(veh_id)
                leader = k.vehicle.get_leader(veh_id)
                headway = k.vehicle.get_headway(veh_id) if leader else None
                
                if veh_id in self.vehicle_history:
                    prev_speed = self.vehicle_history[veh_id][-1].get("speed", speed)
                    dt = 0.1
                    acceleration = (speed - prev_speed) / dt
                else:
                    acceleration = 0.0

                ttc = None
                if leader and headway is not None and headway > 0:
                    leader_speed = k.vehicle.get_speed(leader)
                    relative_speed = speed - leader_speed
                    if relative_speed > 0:
                        ttc = headway / relative_speed
                    else:
                        ttc = float('inf')
                
                veh_record = {
                    "veh_id": veh_id,
                    "speed": speed,
                    "position": position,
                    "lane": lane,
                    "headway": headway if headway is not None else float('inf'),
                    "acceleration": acceleration,
                    "ttc": ttc if ttc is not None else float('inf'),
                    "has_leader": leader is not None,
                }
                
                rl_data[veh_id] = veh_record
                self.vehicle_history[veh_id].append(veh_record)

                agent_idx = rl_ids.index(veh_id)
                agent_id = f"agent_{agent_idx}"
                if agent_id in action_dict:
                    action = action_dict[agent_id]
                    if isinstance(action, (list, np.ndarray)):
                        self.action_history[veh_id].append(action.tolist() if isinstance(action, np.ndarray) else action)
                    else:
                        self.action_history[veh_id].append([float(action)])
                
            except Exception as e:
                continue
        
        step_record["rl_vehicles"] = rl_data
        step_record["rewards"] = reward_dict.copy()
        step_record["actions"] = action_dict.copy()
        
        self.step_data.append(step_record)
    
        for veh_id, data in rl_data.items():
            if data["headway"] < 0:
                self.collisions.append({
                    "step": step,
                    "time": time,
                    "veh_id": veh_id,
                    "headway": data["headway"]
                })
            elif data["headway"] < 2.0:
                self.near_collisions.append({
                    "step": step,
                    "time": time,
                    "veh_id": veh_id,
                    "headway": data["headway"]
                })
    
    def compute_spacing_metrics(self) -> Dict:
        if len(self.step_data) == 0:
            return {}
        
        spacing_history = defaultdict(list)
        
        for step_record in self.step_data:
            for veh_id, veh_data in step_record["rl_vehicles"].items():
                headway = veh_data["headway"]
                if headway != float('inf') and headway > 0:
                    spacing_history[veh_id].append(headway)
        
        if len(spacing_history) == 0:
            return {}
    
        spacing_variances = {}
        for veh_id, spacings in spacing_history.items():
            if len(spacings) > 1:
                spacing_variances[veh_id] = np.var(spacings)
        
        avg_spacing_variance = np.mean(list(spacing_variances.values())) if spacing_variances else 0.0
        oscillation_amplitudes = {}
        for veh_id, spacings in spacing_history.items():
            if len(spacings) > 10:
                oscillation_amplitudes[veh_id] = np.max(spacings) - np.min(spacings)
        
        avg_oscillation_amplitude = np.mean(list(oscillation_amplitudes.values())) if oscillation_amplitudes else 0.0

        damping_ratios = []
        for veh_id, spacings in spacing_history.items():
            if len(spacings) > 20:
                peaks = []
                for i in range(1, len(spacings) - 1):
                    if spacings[i] > spacings[i-1] and spacings[i] > spacings[i+1]:
                        peaks.append(spacings[i])
                
                if len(peaks) >= 2:
                    decay_rates = []
                    for i in range(len(peaks) - 1):
                        if peaks[i] > 0:
                            decay = peaks[i+1] / peaks[i]
                            decay_rates.append(decay)
                    
                    if decay_rates:
                        avg_decay = np.mean(decay_rates)
                        damping_ratios.append(avg_decay)
        
        avg_damping_ratio = np.mean(damping_ratios) if damping_ratios else 1.0
        
        return {
            "spacing_variance_per_vehicle": spacing_variances,
            "avg_spacing_variance": avg_spacing_variance,
            "oscillation_amplitude_per_vehicle": oscillation_amplitudes,
            "avg_oscillation_amplitude": avg_oscillation_amplitude,
            "damping_ratio_per_vehicle": damping_ratios,
            "avg_damping_ratio": avg_damping_ratio,
        }
    
    def compute_efficiency_metrics(self) -> Dict:
        if len(self.step_data) == 0:
            return {}
        
        speeds = []
        for step_record in self.step_data:
            for veh_id, veh_data in step_record["rl_vehicles"].items():
                speed = veh_data["speed"]
                if speed > 0:
                    speeds.append(speed)
        
        if len(speeds) == 0:
            return {}
        
        avg_velocity = np.mean(speeds)
        speed_variance = np.var(speeds)
      
        total_time = self.step_data[-1]["time"] - self.step_data[0]["time"] if len(self.step_data) > 1 else 0.0
        unique_vehicles = set()
        for step_record in self.step_data:
            unique_vehicles.update(step_record["rl_vehicles"].keys())
        
        throughput = len(unique_vehicles) / max(total_time, 1.0) if total_time > 0 else 0.0
        
        return {
            "avg_platoon_velocity": avg_velocity,
            "speed_variance": speed_variance,
            "throughput_vehicles_per_second": throughput,
            "total_vehicles": len(unique_vehicles),
            "total_time": total_time,
        }
    
    def compute_safety_metrics(self) -> Dict:
        collision_count = len(self.collisions)
        near_collision_count = len(self.near_collisions)
        
        min_ttcs = []
        for step_record in self.step_data:
            for veh_id, veh_data in step_record["rl_vehicles"].items():
                ttc = veh_data["ttc"]
                if ttc != float('inf') and ttc > 0:
                    min_ttcs.append(ttc)
        
        min_ttc = min(min_ttcs) if min_ttcs else float('inf')

        max_decelerations = []
        for veh_id, history in self.vehicle_history.items():
            accelerations = [h["acceleration"] for h in history if "acceleration" in h]
            if len(accelerations) > 1:
                jerks = np.diff(accelerations) / 0.1 
                max_decel = min(accelerations) if accelerations else 0.0
                max_decelerations.append(max_decel)
        
        max_deceleration = min(max_decelerations) if max_decelerations else 0.0
        
        return {
            "collision_count": collision_count,
            "near_collision_count": near_collision_count,
            "min_time_to_collision": min_ttc if min_ttc != float('inf') else None,
            "max_deceleration": max_deceleration,
            "collision_details": self.collisions[:10],
        }
    
    def compute_coordination_metrics(self) -> Dict:
        if len(self.action_history) == 0:
            return {}

        action_sequences = {}
        for veh_id, actions in self.action_history.items():
            if len(actions) > 0:
                accel_sequence = [a[0] if isinstance(a, (list, np.ndarray)) and len(a) > 0 else float(a) for a in actions]
                action_sequences[veh_id] = accel_sequence
        
        if len(action_sequences) < 2:
            return {}
        
        correlations = []
        veh_ids = list(action_sequences.keys())
        for i in range(len(veh_ids)):
            for j in range(i + 1, len(veh_ids)):
                seq1 = action_sequences[veh_ids[i]]
                seq2 = action_sequences[veh_ids[j]]
                
                min_len = min(len(seq1), len(seq2))
                if min_len > 10:
                    seq1_aligned = seq1[:min_len]
                    seq2_aligned = seq2[:min_len]
                    corr = np.corrcoef(seq1_aligned, seq2_aligned)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        action_variances = []
        for step_record in self.step_data:
            actions = []
            for veh_id in step_record["rl_vehicles"].keys():
                if veh_id in self.action_history and len(self.action_history[veh_id]) > 0:
                    step_idx = len(self.action_history[veh_id]) - 1
                    if step_idx >= 0:
                        action = self.action_history[veh_id][step_idx]
                        accel = action[0] if isinstance(action, (list, np.ndarray)) and len(action) > 0 else float(action)
                        actions.append(accel)
            
            if len(actions) > 1:
                action_variances.append(np.var(actions))
        
        avg_action_variance = np.mean(action_variances) if action_variances else 0.0
        
        speed_alignments = []
        for step_record in self.step_data:
            speeds = [veh_data["speed"] for veh_data in step_record["rl_vehicles"].values()]
            if len(speeds) > 1:
                cv = np.std(speeds) / max(np.mean(speeds), 1e-3)
                speed_alignments.append(1.0 / (1.0 + cv))
        
        avg_synchronization = np.mean(speed_alignments) if speed_alignments else 0.0
        
        return {
            "action_correlation": avg_correlation,
            "action_correlations_all_pairs": correlations,
            "policy_divergence": avg_action_variance,
            "synchronization_index": avg_synchronization,
        }
    
    def compute_string_stability(self) -> Dict:
        """Compute string stability metric.
        
        String stability: ||e_{i+1}(t)|| / ||e_i(t)|| < 1
        where e_i is the spacing error for vehicle i.
        """
        if len(self.step_data) == 0:
            return {}
        
        target_spacing = 20.0
        spacing_errors = defaultdict(list)
        
        for step_record in self.step_data:
            rl_ids = sorted(step_record["rl_vehicles"].keys())
            for i, veh_id in enumerate(rl_ids):
                headway = step_record["rl_vehicles"][veh_id]["headway"]
                if headway != float('inf') and headway > 0:
                    error = abs(headway - target_spacing)
                    spacing_errors[veh_id].append(error)
        
        if len(spacing_errors) < 2:
            return {}
        
        rl_ids = sorted(spacing_errors.keys())
        string_stability_ratios = []
        
        for i in range(len(rl_ids) - 1):
            veh_i = rl_ids[i]
            veh_i1 = rl_ids[i + 1]
            
            errors_i = spacing_errors[veh_i]
            errors_i1 = spacing_errors[veh_i1]
        
            min_len = min(len(errors_i), len(errors_i1))
            if min_len > 10:
                errors_i_aligned = errors_i[:min_len]
                errors_i1_aligned = errors_i1[:min_len]
                norm_i = np.linalg.norm(errors_i_aligned)
                norm_i1 = np.linalg.norm(errors_i1_aligned)
                
                if norm_i > 1e-6:
                    ratio = norm_i1 / norm_i
                    string_stability_ratios.append(ratio)
        
        avg_ratio = np.mean(string_stability_ratios) if string_stability_ratios else 1.0
        is_string_stable = bool(avg_ratio < 1.0) 
        
        return {
            "string_stability_ratio": float(avg_ratio),
            "is_string_stable": is_string_stable,
            "string_stability_ratios_all_pairs": [float(r) for r in string_stability_ratios],
        }
    
    def compute_all_metrics(self) -> Dict:
        """Compute all metrics."""
        return {
            "policy_type": self.policy_type,
            "spacing_stability": self.compute_spacing_metrics(),
            "efficiency": self.compute_efficiency_metrics(),
            "safety": self.compute_safety_metrics(),
            "coordination": self.compute_coordination_metrics(),
            "string_stability": self.compute_string_stability(),
            "summary": {
                "total_steps": len(self.step_data),
                "total_time": self.step_data[-1]["time"] - self.step_data[0]["time"] if len(self.step_data) > 1 else 0.0,
                "collision_count": len(self.collisions),
            }
        }
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON file."""
        metrics = self.compute_all_metrics()
        
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, np.bool8)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        metrics_serializable = convert_to_serializable(metrics)
        
        with open(filepath, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"Metrics saved to {filepath}")
    
    def save_raw_data(self, filepath: str):
        """Save raw step-by-step data to CSV."""
        if len(self.step_data) == 0:
            return
        
        rows = []
        for step_record in self.step_data:
            base_row = {
                "step": step_record["step"],
                "time": step_record["time"],
                "num_rl_vehicles": step_record["num_rl_vehicles"],
            }
            
            for veh_id, veh_data in step_record["rl_vehicles"].items():
                row = base_row.copy()
                row["veh_id"] = veh_id
                row["speed"] = veh_data["speed"]
                row["position"] = veh_data["position"]
                row["lane"] = veh_data["lane"]
                row["headway"] = veh_data["headway"] if veh_data["headway"] != float('inf') else None
                row["acceleration"] = veh_data["acceleration"]
                row["ttc"] = veh_data["ttc"] if veh_data["ttc"] != float('inf') else None
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"Raw data saved to {filepath}")

