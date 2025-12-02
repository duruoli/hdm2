#!/usr/bin/env python
"""
Example: Train HDM on Metaworld Push Task

This script demonstrates how to train HDM on a Metaworld environment.

Usage:
    python examples/train_metaworld_push.py
    
    # With custom hyperparameters:
    python examples/train_metaworld_push.py --n_cycles 100 --seed 42
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdm import launch


def get_args():
    parser = argparse.ArgumentParser(description='Train HDM on Metaworld Push')
    
    # Environment
    parser.add_argument('--env_name', type=str, default='metaworld_push',
                        help='Environment name')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='experiments/metaworld_push/',
                        help='Directory to save results')
    parser.add_argument('--ckpt_name', type=str, default='',
                        help='Checkpoint name (auto-generated if empty)')
    parser.add_argument('--resume_ckpt', type=str, default='',
                        help='Resume from checkpoint')
    
    # Parallel training
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--num_rollouts_per_mpi', type=int, default=1,
                        help='Rollouts per MPI worker')
    
    # Training schedule
    parser.add_argument('--n_cycles', type=int, default=40,
                        help='Number of training cycles')
    parser.add_argument('--optimize_every', type=int, default=50,
                        help='Optimization frequency')
    parser.add_argument('--n_batches', type=int, default=50,
                        help='Number of optimization batches per update')
    parser.add_argument('--target_update_freq', type=int, default=10,
                        help='Target network update frequency')
    
    # Exploration
    parser.add_argument('--greedy_action', action='store_true', default=True,
                        help='Use greedy actions')
    parser.add_argument('--random_act_prob', type=float, default=0.0,
                        help='Random action probability')
    
    # Replay buffer and HER
    parser.add_argument('--buffer_size', type=int, default=1000000,
                        help='Replay buffer size')
    parser.add_argument('--future_p', type=float, default=0.85,
                        help='HER future sampling probability')
    parser.add_argument('--next_state_p', type=float, default=0.0,
                        help='Next state relabeling probability')
    parser.add_argument('--relabeled_reward_only', action='store_true', default=True,
                        help='Only use relabeled rewards')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    
    # Policy
    parser.add_argument('--lr_actor', type=float, default=5e-4,
                        help='Actor learning rate')
    parser.add_argument('--start_policy_timesteps', type=int, default=1000,
                        help='Timesteps before using learned policy')
    parser.add_argument('--independent_policy', action='store_true', default=True,
                        help='Use independent discrete policy (for large action spaces)')
    
    # Evaluation
    parser.add_argument('--n_initial_rollouts', type=int, default=200,
                        help='Initial random rollouts')
    parser.add_argument('--n_test_rollouts', type=int, default=50,
                        help='Number of test rollouts during evaluation')
    
    # RL
    parser.add_argument('--gamma', type=float, default=0.98,
                        help='Discount factor')
    parser.add_argument('--polyak', type=float, default=0.995,
                        help='Polyak averaging coefficient')
    
    # DQN
    parser.add_argument('--use_dqn', action='store_true', default=True,
                        help='Use DQN')
    parser.add_argument('--double_dqn', action='store_true', default=True,
                        help='Use Double DQN')
    parser.add_argument('--backup_strategy', type=str, default='q_max',
                        choices=['q_max', 'q_softmax', 'q_eps_greedy', 'q_soft_kl'],
                        help='Backup strategy')
    parser.add_argument('--backup_temp', type=float, default=1.0,
                        help='Backup temperature')
    parser.add_argument('--backup_epsilon', type=float, default=0.1,
                        help='Backup epsilon')
    parser.add_argument('--reward_scale', type=float, default=1.0,
                        help='Reward scale')
    parser.add_argument('--reward_bias', type=float, default=-1.0,
                        help='Reward bias')
    parser.add_argument('--targ_clip', action='store_true', default=True,
                        help='Clip target Q-values')
    
    # HDM-specific
    parser.add_argument('--hdm_gamma', type=float, default=0.85,
                        help='HDM discount factor')
    parser.add_argument('--hdm_weights_min', type=float, default=-5.0,
                        help='HDM minimum weight')
    parser.add_argument('--hdm_weights_max', type=float, default=5.0,
                        help='HDM maximum weight')
    parser.add_argument('--hdm_q_coef', type=float, default=1.0,
                        help='HDM Q-loss coefficient')
    parser.add_argument('--hdm_q_normalizer', action='store_true',
                        help='Use HDM Q normalizer')
    parser.add_argument('--hdm_online_o2', action='store_true', default=True,
                        help='Use HDM online o2')
    parser.add_argument('--hdm_backup_strategy', type=str, default='q_max',
                        choices=['q_max', 'q_softmax', 'q_eps_greedy', 'q_soft_kl', 'act_2'],
                        help='HDM backup strategy')
    parser.add_argument('--hdm_bc', action='store_true', default=True,
                        help='Use HDM behavioral cloning')
    parser.add_argument('--hdm_weights_to_indicator', action='store_true', default=True,
                        help='Convert HDM weights to indicators')
    parser.add_argument('--hdm_gamma_use_auto', action='store_true',
                        help='Auto-compute HDM gamma')
    parser.add_argument('--hdm_weights_relabel_mask', action='store_true',
                        help='Use HDM relabeling mask')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    # Set threading environment variables
    n_threads = str(args.n_workers)
    if args.n_workers < 12:
        n_threads = str(12)
    
    os.environ['OMP_NUM_THREADS'] = n_threads
    os.environ['MKL_NUM_THREADS'] = n_threads
    os.environ['IN_MPI'] = n_threads
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    print(f"Training HDM on {args.env_name}")
    print(f"  Seed: {args.seed}")
    print(f"  Cycles: {args.n_cycles}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Independent policy: {args.independent_policy}")
    print()
    
    # Launch training
    algo = launch(args)
    algo.run()
    
    print(f"\nTraining complete! Results saved to {args.save_dir}/{args.ckpt_name}")



