"""Visualize CTDE training results.

This script reads the progress CSV files and creates plots of key metrics.

Usage:
    python visualize_results.py [--results_dir results/ctde]
"""

import os
import sys
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def find_latest_run(results_dir):
    """Find the most recent training run."""
    pattern = os.path.join(results_dir, "PPO_*/PPO_*/progress.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        return None
    
    # Sort by modification time, get most recent
    latest = max(csv_files, key=os.path.getmtime)
    return latest


def load_results(results_dir):
    """Load training results from CSV files."""
    pattern = os.path.join(results_dir, "PPO_*/PPO_*/progress.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No progress.csv files found in {results_dir}")
        return None
    
    # Load the most recent run
    latest_file = max(csv_files, key=os.path.getmtime)
    print(f"Loading results from: {latest_file}")
    
    try:
        df = pd.read_csv(latest_file)
        return df, latest_file
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


def plot_metrics(df, save_path=None):
    """Plot key training metrics."""
    if df is None or len(df) == 0:
        print("No data to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CTDE Training Metrics', fontsize=16, fontweight='bold')
    
    # 1. Episode Reward
    ax = axes[0, 0]
    if 'episode_reward_mean' in df.columns:
        ax.plot(df['training_iteration'], df['episode_reward_mean'], 
                label='Mean Reward', linewidth=2, color='blue')
        if 'episode_reward_min' in df.columns:
            ax.plot(df['training_iteration'], df['episode_reward_min'], 
                    '--', alpha=0.5, label='Min', color='lightblue')
        if 'episode_reward_max' in df.columns:
            ax.plot(df['training_iteration'], df['episode_reward_max'], 
                    '--', alpha=0.5, label='Max', color='lightblue')
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Episode Reward Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Policy Loss
    ax = axes[0, 1]
    # Try different possible column names for policy loss
    policy_loss_cols = [
        'info/learner/shared_policy/learner_stats/policy_loss',
        'policy_loss',
        'info/learner/default_policy/learner_stats/policy_loss',
    ]
    policy_loss_col = None
    for col in policy_loss_cols:
        if col in df.columns:
            policy_loss_col = col
            break
    
    if policy_loss_col:
        ax.plot(df['training_iteration'], df[policy_loss_col], 
                label='Policy Loss', linewidth=2, color='red')
    else:
        ax.text(0.5, 0.5, 'Policy Loss\n(not available)', 
                ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Policy Loss')
    ax.set_title('Policy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Value Loss
    ax = axes[1, 0]
    value_loss_cols = [
        'info/learner/shared_policy/learner_stats/vf_loss',
        'vf_loss',
        'info/learner/default_policy/learner_stats/vf_loss',
    ]
    value_loss_col = None
    for col in value_loss_cols:
        if col in df.columns:
            value_loss_col = col
            break
    
    if value_loss_col:
        ax.plot(df['training_iteration'], df[value_loss_col], 
                label='Value Loss', linewidth=2, color='green')
    else:
        ax.text(0.5, 0.5, 'Value Loss\n(not available)', 
                ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Value Loss')
    ax.set_title('Value Function Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Episode Length
    ax = axes[1, 1]
    if 'episode_len_mean' in df.columns:
        ax.plot(df['training_iteration'], df['episode_len_mean'], 
                label='Mean Length', linewidth=2, color='purple')
    elif 'episodes_this_iter' in df.columns:
        ax.plot(df['training_iteration'], df['episodes_this_iter'], 
                label='Episodes', linewidth=2, color='purple')
    else:
        ax.text(0.5, 0.5, 'Episode Length\n(not available)', 
                ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()
    
    return fig


def print_summary(df):
    """Print summary statistics."""
    if df is None or len(df) == 0:
        return
    
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    
    if 'episode_reward_mean' in df.columns:
        final_reward = df['episode_reward_mean'].iloc[-1]
        max_reward = df['episode_reward_mean'].max()
        print(f"Final Episode Reward: {final_reward:.2f}")
        print(f"Max Episode Reward: {max_reward:.2f}")
    
    print(f"Total Iterations: {len(df)}")
    print(f"Final Iteration: {df['training_iteration'].iloc[-1]}")
    
    if 'episode_len_mean' in df.columns:
        print(f"Mean Episode Length: {df['episode_len_mean'].mean():.1f}")
    
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize CTDE training results")
    parser.add_argument(
        '--results_dir',
        type=str,
        default='../results/ctde',
        help='Path to results directory (default: ../results/ctde)'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save plot to file (e.g., --save training_plot.png)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plot interactively'
    )
    
    args = parser.parse_args()
    
    # Convert relative path to absolute
    if not os.path.isabs(args.results_dir):
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            args.results_dir.lstrip('../')
        )
    else:
        results_dir = args.results_dir
    
    print(f"Looking for results in: {results_dir}")
    
    # Load results
    result = load_results(results_dir)
    if result is None:
        print("No results found!")
        return
    
    df, csv_file = result
    
    # Print summary
    print_summary(df)
    
    # Show available columns
    print("Available metrics:")
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['reward', 'loss', 'entropy', 'length', 'iteration']):
            print(f"  - {col}")
    
    # Plot metrics
    print("\nGenerating plots...")
    plot_metrics(df, save_path=args.save)
    
    if args.show or args.save is None:
        print("\nClose the plot window to exit.")
        plt.show()


if __name__ == "__main__":
    main()

