"""Plot simulation metrics from evaluation runs.

This script reads metrics JSON files and raw CSV data to generate
comprehensive plots for platooning evaluation.
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics")


def find_latest_metrics(policy_type="ctde"):
    """Find the most recent metrics files for a policy type."""
    pattern = os.path.join(METRICS_DIR, f"{policy_type}_metrics_*.json")
    files = glob.glob(pattern)
    if not files:
        return None, None
    
    # Get most recent
    latest_json = max(files, key=os.path.getmtime)
    
    # Find corresponding CSV
    # Extract timestamp from JSON filename (format: {policy_type}_metrics_YYYYMMDD_HHMMSS.json)
    json_basename = os.path.basename(latest_json)
    timestamp = json_basename.replace(f"{policy_type}_metrics_", "").replace(".json", "")
    csv_file = os.path.join(METRICS_DIR, f"{policy_type}_raw_data_{timestamp}.csv")
    
    if not os.path.exists(csv_file):
        csv_file = None
    
    return latest_json, csv_file


def load_metrics(json_path):
    """Load metrics from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_time_series(csv_path, output_path=None, show_plot=True):
    """Plot time-series data from raw CSV."""
    if csv_path is None or not os.path.exists(csv_path):
        print("Warning: No raw CSV data found, skipping time-series plots")
        return
    
    df = pd.read_csv(csv_path)
    
    # Filter to RL vehicles only
    rl_df = df[df['veh_id'].str.startswith('rl_')].copy()
    
    if len(rl_df) == 0:
        print("Warning: No RL vehicle data found in CSV")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Simulation Time-Series Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Speed over time
    ax = axes[0, 0]
    for veh_id in rl_df['veh_id'].unique():
        veh_data = rl_df[rl_df['veh_id'] == veh_id]
        ax.plot(veh_data['time'], veh_data['speed'], label=veh_id, linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Speed (m/s)', fontsize=11)
    ax.set_title('Vehicle Speed Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Headway over time
    ax = axes[0, 1]
    for veh_id in rl_df['veh_id'].unique():
        veh_data = rl_df[rl_df['veh_id'] == veh_id]
        # Filter out None/inf headways
        veh_data_clean = veh_data[veh_data['headway'].notna() & (veh_data['headway'] < 1000)]
        if len(veh_data_clean) > 0:
            ax.plot(veh_data_clean['time'], veh_data_clean['headway'], label=veh_id, linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Headway (m)', fontsize=11)
    ax.set_title('Headway Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Acceleration over time
    ax = axes[1, 0]
    for veh_id in rl_df['veh_id'].unique():
        veh_data = rl_df[rl_df['veh_id'] == veh_id]
        ax.plot(veh_data['time'], veh_data['acceleration'], label=veh_id, linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Acceleration (m/s²)', fontsize=11)
    ax.set_title('Acceleration Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    # Plot 4: Position over time (spacing visualization)
    ax = axes[1, 1]
    for veh_id in rl_df['veh_id'].unique():
        veh_data = rl_df[rl_df['veh_id'] == veh_id]
        ax.plot(veh_data['time'], veh_data['position'], label=veh_id, linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Position (m)', fontsize=11)
    ax.set_title('Vehicle Position Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        time_series_path = output_path.replace('.png', '_time_series.png')
        plt.savefig(time_series_path, dpi=300, bbox_inches='tight')
        print(f"Saved time-series plots to: {time_series_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_summary_metrics(metrics, output_path=None, show_plot=True):
    """Plot summary metrics from JSON."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{metrics["policy_type"].upper()} Simulation Summary Metrics', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Spacing Variance (RL vehicles only)
    ax = axes[0, 0]
    spacing_var = metrics.get("spacing_stability", {}).get("spacing_variance_per_vehicle", {})
    rl_spacing_var = {k: v for k, v in spacing_var.items() if k.startswith('rl_')}
    if rl_spacing_var:
        vehicles = list(rl_spacing_var.keys())
        variances = list(rl_spacing_var.values())
        ax.bar(vehicles, variances, color='steelblue', alpha=0.7)
        ax.set_xlabel('Vehicle ID', fontsize=10)
        ax.set_ylabel('Spacing Variance', fontsize=10)
        ax.set_title('Spacing Variance (RL Vehicles)', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Average Velocity
    ax = axes[0, 1]
    efficiency = metrics.get("efficiency", {})
    avg_velocity = efficiency.get("avg_velocity", 0)
    speed_variance = efficiency.get("speed_variance", 0)
    ax.bar(['Average'], [avg_velocity], color='green', alpha=0.7, label='Mean')
    ax.errorbar(['Average'], [avg_velocity], yerr=[np.sqrt(speed_variance)], 
                fmt='none', color='black', capsize=5, label='±1 std')
    ax.set_ylabel('Velocity (m/s)', fontsize=10)
    ax.set_title('Average Velocity', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Throughput
    ax = axes[0, 2]
    throughput = efficiency.get("throughput", 0)
    ax.bar(['Throughput'], [throughput], color='orange', alpha=0.7)
    ax.set_ylabel('Throughput (veh/s)', fontsize=10)
    ax.set_title('Throughput', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Safety Metrics
    ax = axes[1, 0]
    safety = metrics.get("safety", {})
    collisions = safety.get("collision_count", 0)
    near_collisions = safety.get("near_collision_count", 0)
    min_ttc = safety.get("min_ttc", None)
    
    safety_data = {
        'Collisions': collisions,
        'Near Collisions': near_collisions
    }
    ax.bar(safety_data.keys(), safety_data.values(), color=['red', 'orange'], alpha=0.7)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Safety Metrics', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add min TTC as text
    if min_ttc is not None:
        ax.text(0.5, 0.95, f'Min TTC: {min_ttc:.2f}s', 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 5: String Stability
    ax = axes[1, 1]
    string_stab = metrics.get("string_stability", {})
    stability_ratio = string_stab.get("string_stability_ratio", 1.0)
    is_stable = string_stab.get("is_string_stable", False)
    
    color = 'green' if is_stable else 'red'
    ax.bar(['Ratio'], [stability_ratio], color=color, alpha=0.7)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Stability Threshold')
    ax.set_ylabel('String Stability Ratio', fontsize=10)
    ax.set_title(f'String Stability ({"Stable" if is_stable else "Unstable"})', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Coordination Metrics
    ax = axes[1, 2]
    coord = metrics.get("coordination", {})
    sync_index = coord.get("synchronization_index", 0)
    policy_div = coord.get("policy_divergence", 0)
    
    coord_data = {
        'Sync Index': sync_index,
        'Policy Divergence': policy_div
    }
    ax.bar(coord_data.keys(), coord_data.values(), color=['blue', 'purple'], alpha=0.7)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('Coordination Metrics', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        summary_path = output_path.replace('.png', '_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary plots to: {summary_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot simulation metrics from evaluation runs')
    parser.add_argument('--policy-type', type=str, default='ctde',
                        choices=['ctde', 'independent'],
                        help='Policy type to plot (ctde or independent)')
    parser.add_argument('--json-path', type=str, default=None,
                        help='Path to specific metrics JSON file')
    parser.add_argument('--csv-path', type=str, default=None,
                        help='Path to specific raw data CSV file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path prefix for saved plots')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (only save if --output specified)')
    
    args = parser.parse_args()
    
    # Find metrics files
    if args.json_path:
        json_path = args.json_path
        if not os.path.exists(json_path):
            print(f"Error: JSON file not found: {json_path}")
            sys.exit(1)
        csv_path = args.csv_path
    else:
        json_path, csv_path_default = find_latest_metrics(args.policy_type)
        if json_path is None:
            print(f"Error: No metrics found for policy type: {args.policy_type}")
            sys.exit(1)
        if args.csv_path is None:
            csv_path = csv_path_default
        else:
            csv_path = args.csv_path
    
    print(f"Loading metrics from: {json_path}")
    metrics = load_metrics(json_path)
    
    # Determine output paths
    if args.output:
        output_prefix = args.output
    else:
        output_prefix = f"experiments/{args.policy_type}_simulation_metrics"
    
    # Plot time-series
    if csv_path:
        print(f"Loading raw data from: {csv_path}")
    plot_time_series(csv_path, output_path=output_prefix, show_plot=not args.no_show)
    
    # Plot summary metrics
    plot_summary_metrics(metrics, output_path=output_prefix, show_plot=not args.no_show)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Metrics Summary for {metrics['policy_type'].upper()}:")
    print("=" * 60)
    efficiency = metrics.get("efficiency", {})
    safety = metrics.get("safety", {})
    string_stab = metrics.get("string_stability", {})
    
    print(f"Average Velocity: {efficiency.get('avg_velocity', 0):.2f} m/s")
    print(f"Throughput: {efficiency.get('throughput', 0):.4f} veh/s")
    print(f"Collisions: {safety.get('collision_count', 0)}")
    print(f"Near Collisions: {safety.get('near_collision_count', 0)}")
    print(f"Min TTC: {safety.get('min_ttc', 'N/A')}")
    print(f"String Stability Ratio: {string_stab.get('string_stability_ratio', 0):.2f}")
    print(f"String Stable: {string_stab.get('is_string_stable', False)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

