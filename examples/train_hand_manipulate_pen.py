#!/usr/bin/env python
"""
Example: Train HDM on Gymnasium-Robotics Hand Manipulation Task

This script demonstrates how to train HDM on the challenging Shadow Hand pen manipulation task.

Usage:
    python examples/train_hand_manipulate_pen.py
    
    # With custom hyperparameters:
    python examples/train_hand_manipulate_pen.py --n_cycles 100 --n_initial_rollouts 500
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdm import launch


def get_args():
    parser = argparse.ArgumentParser(description='Train HDM on Hand Manipulation Pen')
    
    # Environment
    parser.add_argument('--env_name', type=str, default='hand_manipulate_pen',
                        help='Environment name')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='experiments/hand_manipulate_pen/')
    parser.add_argument('--ckpt_name', type=str, default='')
    parser.add_argument('--resume_ckpt', type=str, default='')
    
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--num_rollouts_per_mpi', type=int, default=1)
    
    # Training - Hand manipulation is harder, so increase cycles
    parser.add_argument('--n_cycles', type=int, default=100,
                        help='Number of training cycles (increased for harder task)')
    parser.add_argument('--optimize_every', type=int, default=50)
    parser.add_argument('--n_batches', type=int, default=50)
    parser.add_argument('--target_update_freq', type=int, default=10)
    
    parser.add_argument('--greedy_action', action='store_true', default=True)
    parser.add_argument('--random_act_prob', type=float, default=0.0)
    
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--future_p', type=float, default=0.85)
    parser.add_argument('--next_state_p', type=float, default=0.0)
    parser.add_argument('--relabeled_reward_only', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=256)
    
    parser.add_argument('--lr_actor', type=float, default=5e-4)
    parser.add_argument('--start_policy_timesteps', type=int, default=1000)
    parser.add_argument('--independent_policy', action='store_true', default=True)
    
    # Evaluation - Increase initial rollouts for exploration
    parser.add_argument('--n_initial_rollouts', type=int, default=500,
                        help='Initial random rollouts (increased for harder task)')
    parser.add_argument('--n_test_rollouts', type=int, default=50)
    
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--polyak', type=float, default=0.995)
    
    parser.add_argument('--use_dqn', action='store_true', default=True)
    parser.add_argument('--double_dqn', action='store_true', default=True)
    parser.add_argument('--backup_strategy', type=str, default='q_max',
                        choices=['q_max', 'q_softmax', 'q_eps_greedy', 'q_soft_kl'])
    parser.add_argument('--backup_temp', type=float, default=1.0)
    parser.add_argument('--backup_epsilon', type=float, default=0.1)
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--reward_bias', type=float, default=-1.0)
    parser.add_argument('--targ_clip', action='store_true', default=True)
    
    parser.add_argument('--hdm_gamma', type=float, default=0.85)
    parser.add_argument('--hdm_weights_min', type=float, default=-5.0)
    parser.add_argument('--hdm_weights_max', type=float, default=5.0)
    parser.add_argument('--hdm_q_coef', type=float, default=1.0)
    parser.add_argument('--hdm_q_normalizer', action='store_true')
    parser.add_argument('--hdm_online_o2', action='store_true', default=True)
    parser.add_argument('--hdm_backup_strategy', type=str, default='q_max',
                        choices=['q_max', 'q_softmax', 'q_eps_greedy', 'q_soft_kl', 'act_2'])
    parser.add_argument('--hdm_bc', action='store_true', default=True)
    parser.add_argument('--hdm_weights_to_indicator', action='store_true', default=True)
    parser.add_argument('--hdm_gamma_use_auto', action='store_true')
    parser.add_argument('--hdm_weights_relabel_mask', action='store_true')
    
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
    
    print("=" * 60)
    print("Training HDM on Shadow Hand Pen Manipulation")
    print("=" * 60)
    print(f"Environment: {args.env_name}")
    print(f"Seed: {args.seed}")
    print(f"Training cycles: {args.n_cycles}")
    print(f"Initial rollouts: {args.n_initial_rollouts}")
    print(f"Save directory: {args.save_dir}")
    print(f"Independent policy: {args.independent_policy}")
    print()
    print("Note: Hand manipulation is a challenging task.")
    print("      Training may take longer than simpler environments.")
    print("=" * 60)
    print()
    
    # Launch training
    algo = launch(args)
    algo.run()
    
    print()
    print("=" * 60)
    print(f"Training complete! Results saved to {args.save_dir}/{args.ckpt_name}")
    print("=" * 60)


