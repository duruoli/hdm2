#!/usr/bin/env python3
"""
Visualize training progress from CSV file
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def plot_training_metrics(csv_path, output_dir=None):
    """
    Plot key training metrics from progress CSV
    
    Args:
        csv_path: Path to progress.csv file
        output_dir: Directory to save plots (defaults to same directory as CSV)
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Training Progress Visualization', fontsize=16, fontweight='bold')
    
    # 1. Success Rates
    ax = axes[0, 0]
    ax.plot(df['Epoch'], df['TestSuccessRate'], label='Test Success Rate', linewidth=2)
    ax.plot(df['Epoch'], df['TestBestSuccessRate'], label='Test Best Success Rate', linewidth=2, alpha=0.7)
    ax.plot(df['Epoch'], df['TrainSuccess'], label='Train Success', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rates Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # 2. Losses
    ax = axes[0, 1]
    ax.plot(df['Epoch'], df['Loss_q'], label='Q Loss', linewidth=2)
    ax.plot(df['Epoch'], df['Loss_her'], label='HER Loss', linewidth=2, alpha=0.7)
    ax.plot(df['Epoch'], df['Loss_hdm'], label='HDM Loss', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visibility
    
    # 3. Test Final Distance
    ax = axes[0, 2]
    ax.plot(df['Epoch'], df['TestFinalDist'], label='Test Final Distance', linewidth=2, color='purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Distance')
    ax.set_title('Test Final Distance to Goal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Q-values
    ax = axes[1, 0]
    ax.plot(df['Epoch'], df['q_o2_max'], label='Q Max (o2)', linewidth=2)
    ax.plot(df['Epoch'], df['q_o2_soft'], label='Q Soft (o2)', linewidth=2, alpha=0.7)
    ax.plot(df['Epoch'], df['q_targ'], label='Q Target', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Q-value')
    ax.set_title('Q-values Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Rewards
    ax = axes[1, 1]
    ax.plot(df['Epoch'], df['reward'], label='Reward', linewidth=2, color='green')
    ax.plot(df['Epoch'], df['exp_reward_avg'], label='Exp Reward Avg', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.set_title('Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. HDM Gamma
    ax = axes[1, 2]
    ax.plot(df['Epoch'], df['hdm_gamma'], label='HDM Gamma', linewidth=2, color='orange')
    ax.plot(df['Epoch'], df['hdm_gamma_auto'], label='HDM Gamma Auto', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gamma')
    ax.set_title('HDM Gamma Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Agent Change Ratios
    ax = axes[2, 0]
    ax.plot(df['Epoch'], df['TestAgChangeRatio'], label='Test', linewidth=2)
    ax.plot(df['Epoch'], df['TrainAgChangeRatio'], label='Train', linewidth=2, alpha=0.7)
    ax.plot(df['Epoch'], df['Inner_Test_AgChangeRatio'], label='Inner Test', linewidth=2, alpha=0.7)
    ax.plot(df['Epoch'], df['Inner_Train_AgChangeRatio'], label='Inner Train', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Change Ratio')
    ax.set_title('Agent Change Ratios')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 8. Time Metrics
    ax = axes[2, 1]
    ax.plot(df['Epoch'], df['TimePerTrainIter'], label='Time per Train Iter', linewidth=2)
    ax.plot(df['Epoch'], df['TimePerSeqRollout'], label='Time per Rollout', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time (s)')
    ax.set_title('Training Time Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 9. Replay Buffer
    ax = axes[2, 2]
    ax.plot(df['Epoch'], df['replay_fill_ratio'], label='Fill Ratio', linewidth=2, color='brown')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Fill Ratio')
    ax.set_title('Replay Buffer Fill Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / 'training_progress.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to: {output_path}")
    
    # Create a second figure for detailed loss analysis
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Detailed Loss and Q-value Analysis', fontsize=16, fontweight='bold')
    
    # Loss comparison (linear scale)
    ax = axes2[0, 0]
    ax.plot(df['Epoch'], df['Loss_q'], label='Q Loss', linewidth=2)
    ax.plot(df['Epoch'], df['Loss_her'], label='HER Loss', linewidth=2, alpha=0.7)
    ax.plot(df['Epoch'], df['Loss_hdm'], label='HDM Loss', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses (Linear Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # All Q-values
    ax = axes2[0, 1]
    ax.plot(df['Epoch'], df['q_bg'], label='Q BG', linewidth=1.5, alpha=0.8)
    ax.plot(df['Epoch'], df['q_o2_a2'], label='Q O2 A2', linewidth=1.5, alpha=0.8)
    ax.plot(df['Epoch'], df['q_o2_max'], label='Q O2 Max', linewidth=1.5, alpha=0.8)
    ax.plot(df['Epoch'], df['q_targ'], label='Q Target', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Q-value')
    ax.set_title('All Q-values Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # HDM Q-values
    ax = axes2[1, 0]
    ax.plot(df['Epoch'], df['hdm_q_o1'], label='HDM Q O1', linewidth=2)
    ax.plot(df['Epoch'], df['hdm_q_o2'], label='HDM Q O2', linewidth=2, alpha=0.7)
    ax.plot(df['Epoch'], df['hdm_q_to_minimize'], label='HDM Q to Minimize', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Q-value')
    ax.set_title('HDM Q-values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Statistics summary
    ax = axes2[1, 1]
    ax.axis('off')
    
    # Calculate summary statistics
    stats_text = f"""
    Training Summary Statistics
    ═══════════════════════════════
    
    Success Rates:
    • Final Test Success: {df['TestSuccessRate'].iloc[-1]:.3f}
    • Mean Test Success: {df['TestSuccessRate'].mean():.3f}
    • Std Test Success: {df['TestSuccessRate'].std():.3f}
    
    Final Distance:
    • Final: {df['TestFinalDist'].iloc[-1]:.4f}
    • Mean: {df['TestFinalDist'].mean():.4f}
    • Min: {df['TestFinalDist'].min():.4f}
    
    Losses (Final):
    • Q Loss: {df['Loss_q'].iloc[-1]:.2e}
    • HER Loss: {df['Loss_her'].iloc[-1]:.2e}
    • HDM Loss: {df['Loss_hdm'].iloc[-1]:.2e}
    
    Training Progress:
    • Total Epochs: {len(df)}
    • Total Steps: {df['TotalTimeSteps'].iloc[-1]:.0f}
    • Total Time: {df['Time'].iloc[-1]/3600:.2f} hours
    • Replay Fill: {df['replay_fill_ratio'].iloc[-1]:.2f}
    """
    
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the second plot
    output_path2 = output_dir / 'training_analysis.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✅ Saved analysis to: {output_path2}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total Epochs: {len(df)}")
    print(f"Total Steps: {df['TotalTimeSteps'].iloc[-1]:.0f}")
    print(f"Total Time: {df['Time'].iloc[-1]/3600:.2f} hours")
    print(f"\nFinal Metrics:")
    print(f"  • Test Success Rate: {df['TestSuccessRate'].iloc[-1]:.3f}")
    print(f"  • Test Final Distance: {df['TestFinalDist'].iloc[-1]:.4f}")
    print(f"  • Q Loss: {df['Loss_q'].iloc[-1]:.2e}")
    print(f"\nKey Observations:")
    
    # Analysis
    if df['TestSuccessRate'].std() < 0.01:
        print("  ⚠️  Test success rate is constant (std < 0.01)")
        print("     This suggests the agent achieved perfect performance early")
        print("     or there may be an issue with the evaluation.")
    
    if df['TestFinalDist'].iloc[-1] < 0.05:
        print("  ✅ Agent reaches very close to the goal (dist < 0.05)")
    
    if df['Loss_q'].iloc[-1] < 1e-6:
        print("  ℹ️  Q-loss is very small, suggesting convergence")
    
    print("="*60 + "\n")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize training progress from CSV')
    parser.add_argument('csv_path', type=str, nargs='?', 
                       default='experiments/metaworld_door_open/progress.csv',
                       help='Path to progress.csv file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save plots (default: same as CSV)')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        return
    
    print(f"📊 Visualizing training from: {csv_path}")
    plot_training_metrics(csv_path, args.output_dir)

if __name__ == '__main__':
    main()
