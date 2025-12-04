"""Plot training metrics from Ray Tune progress.csv files.

This script reads training progress CSV files and generates plots
for key metrics like episode reward, policy loss, value function loss, etc.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def find_latest_run(results_subdir="ctde"):
    """Find the most recent training run in the results directory."""
    results_path = os.path.join(RESULTS_DIR, results_subdir)
    if not os.path.exists(results_path):
        return None
    
    # Find all experiment directories
    exp_dirs = []
    for item in os.listdir(results_path):
        item_path = os.path.join(results_path, item)
        if os.path.isdir(item_path) and item.startswith("PPO_"):
            exp_dirs.append(item_path)
    
    if not exp_dirs:
        return None
    
    # Sort by modification time, get most recent
    exp_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_dir = exp_dirs[0]
    
    # Find progress.csv in the subdirectory
    for root, dirs, files in os.walk(latest_dir):
        if "progress.csv" in files:
            return os.path.join(root, "progress.csv")
    
    return None


def load_progress_csv(csv_path):
    """Load and parse Ray Tune progress.csv file."""
    df = pd.read_csv(csv_path)
    
    # Parse list columns (like hist_stats/episode_reward)
    for col in df.columns:
        if df[col].dtype == object:
            # Try to parse as list
            try:
                # Remove brackets and quotes, split by comma
                df[col] = df[col].apply(
                    lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x
                )
            except:
                pass
    
    return df


def plot_training_metrics(df, output_path=None, show_plot=True):
    """Plot key training metrics from progress CSV.
    
    Creates Figure 1: Training Curves
    - Episode Reward (main plot)
    - KL Divergence (small subplot)
    - Entropy (subplot)
    """
    
    # Create figure with custom layout: main plot + 2 subplots
    fig = plt.figure(figsize=(12, 8))
    
    # Main plot: Episode Reward (takes up most of the space)
    ax_main = plt.subplot(2, 1, 1)
    if 'episode_reward_mean' in df.columns:
        ax_main.plot(df['training_iteration'], df['episode_reward_mean'], 
                     label='Mean', linewidth=2.5, color='blue')
    if 'episode_reward_max' in df.columns:
        ax_main.plot(df['training_iteration'], df['episode_reward_max'], 
                     label='Max', linewidth=1.5, alpha=0.7, color='green', linestyle='--')
    if 'episode_reward_min' in df.columns:
        ax_main.plot(df['training_iteration'], df['episode_reward_min'], 
                     label='Min', linewidth=1.5, alpha=0.7, color='red', linestyle='--')
    ax_main.set_xlabel('Training Iteration', fontsize=12)
    ax_main.set_ylabel('Episode Reward', fontsize=12)
    ax_main.set_title('Episode Reward', fontsize=14, fontweight='bold')
    ax_main.legend(fontsize=10)
    ax_main.grid(True, alpha=0.3)
    
    # Subplot 1: KL Divergence (small subplot)
    ax_kl = plt.subplot(2, 2, 3)
    # Try CTDE format first (shared_policy), then independent format (agent_0, agent_1, etc.)
    kl_col = None
    if 'info/learner/shared_policy/learner_stats/kl' in df.columns:
        kl_col = 'info/learner/shared_policy/learner_stats/kl'
    else:
        # Find first agent's KL (for independent training)
        for col in df.columns:
            if 'info/learner/agent_' in col and '/learner_stats/kl' in col:
                kl_col = col
                break
    
    if kl_col and kl_col in df.columns:
        kl = df[kl_col]
        ax_kl.plot(df['training_iteration'], kl, 
                   linewidth=2, color='brown')
        ax_kl.set_xlabel('Training Iteration', fontsize=10)
        ax_kl.set_ylabel('KL Divergence', fontsize=10)
        ax_kl.set_title('KL Divergence', fontsize=11, fontweight='bold')
        ax_kl.grid(True, alpha=0.3)
    
    # Subplot 2: Entropy
    ax_entropy = plt.subplot(2, 2, 4)
    # Try CTDE format first (shared_policy), then independent format (agent_0, agent_1, etc.)
    entropy_col = None
    if 'info/learner/shared_policy/learner_stats/entropy' in df.columns:
        entropy_col = 'info/learner/shared_policy/learner_stats/entropy'
    else:
        # Find first agent's entropy (for independent training)
        for col in df.columns:
            if 'info/learner/agent_' in col and '/learner_stats/entropy' in col:
                entropy_col = col
                break
    
    if entropy_col and entropy_col in df.columns:
        entropy = df[entropy_col]
        ax_entropy.plot(df['training_iteration'], entropy, 
                        linewidth=2, color='teal')
        ax_entropy.set_xlabel('Training Iteration', fontsize=10)
        ax_entropy.set_ylabel('Entropy', fontsize=10)
        ax_entropy.set_title('Policy Entropy', fontsize=11, fontweight='bold')
        ax_entropy.grid(True, alpha=0.3)
    
    # Determine title based on which format we found
    if 'info/learner/shared_policy' in str(df.columns):
        title = 'CTDE Training Curves'
    else:
        title = 'Independent Training Curves'
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from Ray Tune progress.csv')
    parser.add_argument('--results-dir', type=str, default='ctde',
                        help='Results subdirectory (ctde, independent, multi_agent)')
    parser.add_argument('--csv-path', type=str, default=None,
                        help='Path to specific progress.csv file (overrides --results-dir)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for saved plot (e.g., training_plots.png)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plot (only save if --output specified)')
    
    args = parser.parse_args()
    
    # Find progress.csv file
    if args.csv_path:
        csv_path = args.csv_path
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)
    else:
        csv_path = find_latest_run(args.results_dir)
        if csv_path is None:
            print(f"Error: No training runs found in {os.path.join(RESULTS_DIR, args.results_dir)}")
            sys.exit(1)
    
    print(f"Loading training metrics from: {csv_path}")
    
    # Load and plot
    df = load_progress_csv(csv_path)
    print(f"Loaded {len(df)} training iterations")
    print(f"Training iterations: {df['training_iteration'].min():.0f} to {df['training_iteration'].max():.0f}")
    
    if 'episode_reward_mean' in df.columns:
        final_reward = df['episode_reward_mean'].iloc[-1]
        max_reward = df['episode_reward_mean'].max()
        print(f"Final episode reward (mean): {final_reward:.2f}")
        print(f"Max episode reward (mean): {max_reward:.2f}")
    
    plot_training_metrics(df, output_path=args.output, show_plot=not args.no_show)


if __name__ == "__main__":
    main()

