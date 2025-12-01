"""
Test script for Metaworld and Gymnasium-Robotics integration with HDM.

This script demonstrates how to:
1. Create environments from Metaworld and Gymnasium-Robotics
2. Verify HDM compatibility (all required methods and spaces)
3. Run quick training tests

Usage:
    # Test all available environments
    python test_external_envs_integration.py
    
    # Test specific environment
    python test_external_envs_integration.py --env metaworld_push
    
    # Run training test
    python test_external_envs_integration.py --env metaworld_push --train
"""

import argparse
import numpy as np
import gym


def test_hdm_interface(env, env_name):
    """Test that environment has all required HDM interface methods and spaces."""
    print(f"\n{'='*60}")
    print(f"Testing: {env_name}")
    print(f"{'='*60}")
    
    errors = []
    
    # Check required spaces
    required_spaces = ['observation_space', 'goal_space', 'state_space', 'action_space']
    for space_name in required_spaces:
        if not hasattr(env, space_name):
            errors.append(f"Missing space: {space_name}")
        else:
            space = getattr(env, space_name)
            print(f"✓ {space_name}: {space}")
    
    # Check required methods
    required_methods = [
        'reset', 'step', 'observation', 'extract_goal', 
        'sample_goal', 'goal_distance', 'seed'
    ]
    for method_name in required_methods:
        if not hasattr(env, method_name):
            errors.append(f"Missing method: {method_name}")
        else:
            print(f"✓ Method: {method_name}")
    
    # Test reset
    print("\n--- Testing reset() ---")
    state = env.reset()
    print(f"State shape: {state.shape}")
    expected_shape = (
        env.observation_space.shape[0] + env.goal_space.shape[0],
    )
    if state.shape != expected_shape:
        errors.append(
            f"Reset returned wrong shape: {state.shape}, expected {expected_shape}"
        )
    else:
        print(f"✓ State shape matches (obs_dim + goal_dim): {expected_shape}")
    
    # Test observation extraction
    print("\n--- Testing observation() ---")
    obs = env.observation(state)
    print(f"Observation shape: {obs.shape}")
    if obs.shape != env.observation_space.shape:
        errors.append(
            f"observation() returned wrong shape: {obs.shape}, "
            f"expected {env.observation_space.shape}"
        )
    else:
        print(f"✓ Observation shape correct: {obs.shape}")
    
    # Test goal extraction
    print("\n--- Testing extract_goal() ---")
    goal = env.extract_goal(state)
    print(f"Goal shape: {goal.shape}")
    if goal.shape != env.goal_space.shape:
        errors.append(
            f"extract_goal() returned wrong shape: {goal.shape}, "
            f"expected {env.goal_space.shape}"
        )
    else:
        print(f"✓ Goal shape correct: {goal.shape}")
    
    # Test step
    print("\n--- Testing step() ---")
    action = env.action_space.sample()
    result = env.step(action)
    if len(result) != 4:
        errors.append(
            f"step() returned {len(result)} values, expected 4 (old gym API)"
        )
    else:
        next_state, reward, done, info = result
        print(f"✓ step() returns 4 values (old gym API)")
        print(f"  Next state shape: {next_state.shape}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        print(f"  Info keys: {list(info.keys())}")
        
        if next_state.shape != expected_shape:
            errors.append(
                f"step() returned state with wrong shape: {next_state.shape}"
            )
    
    # Test sample_goal
    print("\n--- Testing sample_goal() ---")
    sampled_goal_state = env.sample_goal()
    print(f"Sampled goal state shape: {sampled_goal_state.shape}")
    if sampled_goal_state.shape != expected_shape:
        errors.append(
            f"sample_goal() returned wrong shape: {sampled_goal_state.shape}"
        )
    else:
        print(f"✓ sample_goal() shape correct: {sampled_goal_state.shape}")
    
    # Test goal_distance
    print("\n--- Testing goal_distance() ---")
    state1 = env.reset()
    state2 = env.sample_goal()
    distance = env.goal_distance(state1, state2)
    print(f"Distance between two states: {distance}")
    if not isinstance(distance, (float, np.floating, np.ndarray)):
        errors.append(f"goal_distance() returned invalid type: {type(distance)}")
    else:
        print(f"✓ goal_distance() returns numeric value")
    
    # Test seed
    print("\n--- Testing seed() ---")
    env.seed(42)
    print(f"✓ seed() callable")
    
    # Print summary
    print(f"\n{'='*60}")
    if errors:
        print(f"❌ FAILED: {len(errors)} error(s) found:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print(f"✅ PASSED: All tests passed for {env_name}")
        return True


def test_rollout(env, env_name, num_steps=10):
    """Test a short rollout in the environment."""
    print(f"\n{'='*60}")
    print(f"Testing rollout: {env_name}")
    print(f"{'='*60}")
    
    state = env.reset()
    total_reward = 0
    
    for step in range(num_steps):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"Episode finished at step {step + 1}")
            state = env.reset()
    
    print(f"✓ Completed {num_steps} steps")
    print(f"  Total reward: {total_reward}")
    print(f"  Final state shape: {state.shape}")
    
    return True


def run_training_test(env_name, num_epochs=2):
    """Run a minimal HDM training test."""
    print(f"\n{'='*60}")
    print(f"Running training test: {env_name}")
    print(f"{'='*60}")
    
    try:
        import modern_envs as envs
        from hdm import launch
        
        # Create a minimal args object
        class Args:
            def __init__(self):
                self.env_name = env_name
                self.seed = 0
                self.save_dir = 'experiments/test_external_envs/'
                self.ckpt_name = f'test_{env_name}'
                self.resume_ckpt = ''
                
                self.n_workers = 1
                self.num_rollouts_per_mpi = 1
                
                self.n_cycles = num_epochs
                self.optimize_every = 10
                self.n_batches = 10
                self.target_update_freq = 5
                
                self.greedy_action = True
                self.random_act_prob = 0.0
                
                self.buffer_size = 10000
                self.future_p = 0.85
                self.next_state_p = 0.0
                self.relabeled_reward_only = True
                self.batch_size = 32
                
                self.lr_actor = 5e-4
                self.start_policy_timesteps = 100
                
                self.n_initial_rollouts = 10
                self.n_test_rollouts = 5
                
                self.gamma = 0.98
                self.polyak = 0.995
                
                self.independent_policy = True  # Use for large action spaces
                
                self.use_dqn = True
                self.double_dqn = True
                self.backup_strategy = 'q_max'
                self.backup_temp = 1.0
                self.backup_epsilon = 0.1
                self.reward_scale = 1.0
                self.reward_bias = -1.0
                self.targ_clip = True
                
                self.hdm_gamma = 0.85
                self.hdm_weights_min = -5.0
                self.hdm_weights_max = 5.0
                self.hdm_q_coef = 1.0
                self.hdm_q_normalizer = False
                self.hdm_online_o2 = True
                self.hdm_backup_strategy = 'q_max'
                self.hdm_bc = True
                self.hdm_weights_to_indicator = True
                self.hdm_gamma_use_auto = False
                self.hdm_weights_relabel_mask = False
        
        args = Args()
        
        print(f"\n✓ Launching HDM training with {num_epochs} epochs...")
        algo = launch(args)
        algo.run()
        
        print(f"\n✅ Training test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Test Metaworld and Gymnasium-Robotics integration with HDM'
    )
    parser.add_argument(
        '--env',
        type=str,
        default=None,
        help='Specific environment to test (default: test all)'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Run training test (requires HDM)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='Number of training epochs for training test'
    )
    args = parser.parse_args()
    
    # Import after parsing args so we can provide helpful error messages
    try:
        import modern_envs as envs
    except ImportError:
        print("❌ modern_envs not found. Make sure you're in the correct directory.")
        return
    
    # Determine which environments to test
    if args.env:
        env_names = [args.env]
    else:
        # Test selection of environments from each source
        env_names = []
        
        # Native environments (skip these, they're already tested)
        # env_names.extend(['pusher', 'door'])
        
        # Metaworld environments
        if envs.METAWORLD_AVAILABLE:
            env_names.extend(['metaworld_push', 'metaworld_reach'])
        else:
            print("⚠️  Metaworld not available (install: pip install metaworld)")
        
        # Gymnasium-Robotics environments (only stable ones)
        if envs.GYMNASIUM_ROBOTICS_AVAILABLE:
            # Use Fetch and PointMaze (confirmed working)
            # Note: Hand manipulation environments have MuJoCo compatibility issues
            env_names.extend(['fetch_push', 'pointmaze_medium'])
        else:
            print("⚠️  Gymnasium-Robotics not available (install: pip install gymnasium-robotics)")
    
    if not env_names:
        print("❌ No environments to test. Install metaworld or gymnasium-robotics.")
        return
    
    print(f"\n{'='*60}")
    print(f"Testing {len(env_names)} environment(s)")
    print(f"{'='*60}")
    
    results = {}
    
    for env_name in env_names:
        try:
            # Create environment
            env = envs.create_env(env_name)
            env_params = envs.get_env_params(env_name)
            
            print(f"\n✓ Created environment: {env_name}")
            print(f"  Parameters: {env_params}")
            
            # Test HDM interface
            interface_ok = test_hdm_interface(env, env_name)
            
            # Test rollout
            rollout_ok = test_rollout(env, env_name, num_steps=10)
            
            results[env_name] = interface_ok and rollout_ok
            
            env.close()
            
        except Exception as e:
            print(f"\n❌ Error testing {env_name}: {e}")
            import traceback
            traceback.print_exc()
            results[env_name] = False
    
    # Run training test if requested
    if args.train:
        for env_name in env_names:
            if results.get(env_name, False):
                train_ok = run_training_test(env_name, num_epochs=args.epochs)
                results[env_name] = results[env_name] and train_ok
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for ok in results.values() if ok)
    total = len(results)
    
    for env_name, ok in results.items():
        status = "✅ PASSED" if ok else "❌ FAILED"
        print(f"{status}: {env_name}")
    
    print(f"\n{passed}/{total} environments passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == '__main__':
    main()

