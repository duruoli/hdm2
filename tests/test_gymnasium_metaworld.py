#!/usr/bin/env python3
"""
Test script to explore Gymnasium-Robotics and Metaworld environments
for goal-conditioned RL with HDM project.

This script tests:
1. Metaworld environments (for push, door, etc.)
2. Gymnasium-Robotics environments (for robotic manipulation tasks)
3. Compatibility with goal-conditioned RL (observation/goal structure)
"""

import numpy as np
import gymnasium as gym
import torch
import random


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_metaworld():
    """Test Metaworld environments."""
    print_section("METAWORLD ENVIRONMENTS")
    
    try:
        import metaworld
        print(f"✓ Metaworld imported successfully")
        
        # Get all available benchmark environments
        print("\n--- Available Metaworld Benchmarks ---")
        mt1 = metaworld.MT1('pick-place-v3')
        print("MT1 (single task) benchmark available")
        
        # List some key tasks we might be interested in
        print("\n--- Key Metaworld Tasks for HDM ---")
        relevant_tasks = [
            'push-v3',          # Push task (similar to our Sawyer Push)
            'door-open-v3',     # Door task (similar to our Sawyer Door)
            'door-close-v3',    # Door closing
            'drawer-open-v3',   # Drawer manipulation
            'drawer-close-v3',
            'reach-v3',         # Simple reaching
            'pick-place-v3',    # Pick and place
        ]
        
        available_tasks = []
        for task_name in relevant_tasks:
            try:
                mt1_task = metaworld.MT1(task_name)
                available_tasks.append(task_name)
                print(f"  ✓ {task_name}")
            except Exception as e:
                print(f"  ✗ {task_name} - {str(e)[:50]}")
        
        # Test a specific environment in detail
        if available_tasks:
            test_task = 'push-v3'  # Test push task first
            print(f"\n--- Testing {test_task} in detail ---")
            
            mt1 = metaworld.MT1(test_task)
            env = mt1.train_classes[test_task]()
            
            # Get a random task (goal)
            tasks = mt1.train_tasks
            task = random.sample(tasks, 1)[0]
            env.set_task(task)
            
            print(f"Environment: {env}")
            print(f"Number of training tasks: {len(mt1.train_tasks)}")
            print(f"Number of test tasks: {len(mt1.test_tasks)}")
            
            # Test observation and action spaces
            obs, info = env.reset()
            print(f"\nObservation space: {env.observation_space}")
            print(f"Action space: {env.action_space}")
            print(f"Initial observation shape: {obs.shape}")
            print(f"Initial observation sample: {obs[:5]}...")
            
            # Check if goal is in the observation or info
            print(f"\nInfo keys: {info.keys()}")
            
            # Check observation structure
            print(f"\nObservation structure analysis:")
            print(f"  - Full obs length: {len(obs)}")
            
            # Try to understand observation structure
            # Metaworld typically includes: [hand_pos(3), gripper_state(1), object_pos(3), goal_pos(3), ...]
            if len(obs) >= 39:  # Standard metaworld obs
                print(f"  - Hand position (first 3): {obs[:3]}")
                print(f"  - Object positions and states: {obs[3:].shape}")
            
            # Test stepping through environment
            print(f"\n--- Testing environment dynamics ---")
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Action: {action}")
            print(f"Reward: {reward}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(f"Info keys: {info.keys()}")
            if 'success' in info:
                print(f"Success metric available: {info['success']}")
            
            # Run a few steps
            print(f"\n--- Running 10 random steps ---")
            obs, _ = env.reset()
            rewards = []
            for i in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                if terminated or truncated:
                    obs, _ = env.reset()
            
            print(f"Rewards over 10 steps: {rewards}")
            print(f"Mean reward: {np.mean(rewards):.4f}")
            
            return True
            
    except Exception as e:
        print(f"✗ Error testing Metaworld: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gymnasium_robotics():
    """Test Gymnasium-Robotics environments."""
    print_section("GYMNASIUM-ROBOTICS ENVIRONMENTS")
    
    try:
        import gymnasium_robotics
        print(f"✓ Gymnasium-Robotics installed")
        
        # List relevant environments (using v4 as v3 is deprecated)
        print("\n--- Relevant Gymnasium-Robotics Environments ---")
        relevant_envs = [
            'FetchPush-v4',
            'FetchSlide-v4',
            'FetchReach-v4',
            'FetchPickAndPlace-v4',
            'HandReach-v3',
        ]
        
        available_envs = []
        for env_name in relevant_envs:
            try:
                env = gym.make(env_name)
                available_envs.append(env_name)
                print(f"  ✓ {env_name}")
                env.close()
            except Exception as e:
                print(f"  ✗ {env_name} - {str(e)[:50]}")
        
        # Test a specific environment in detail
        if available_envs:
            test_env = 'FetchPush-v4'
            print(f"\n--- Testing {test_env} in detail ---")
            
            env = gym.make(test_env)
            
            print(f"Environment: {env}")
            print(f"Observation space: {env.observation_space}")
            print(f"Action space: {env.action_space}")
            
            # Reset and inspect observation structure
            obs, info = env.reset()
            
            print(f"\n--- Goal-Conditioned Structure ---")
            if isinstance(obs, dict):
                print("✓ Observation is a dictionary (goal-conditioned format!)")
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        print(f"  - {key}: shape {value.shape}, dtype {value.dtype}")
                        print(f"    Sample: {value.flatten()[:5]}...")
                    else:
                        print(f"  - {key}: {type(value)}")
                
                # Check for standard goal-conditioned keys
                if 'observation' in obs and 'desired_goal' in obs and 'achieved_goal' in obs:
                    print("\n✓ Standard goal-conditioned RL format detected!")
                    print(f"  - observation: agent state")
                    print(f"  - desired_goal: target goal") 
                    print(f"  - achieved_goal: current achieved goal")
                    
                    # This is exactly what we need for HDM!
                    print("\n✓✓ This format is PERFECT for HDM/HER training!")
            else:
                print(f"✗ Observation is not a dict: {type(obs)}")
                print(f"  Shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
            
            # Test stepping
            print(f"\n--- Testing environment dynamics ---")
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Reward: {reward}")
            print(f"Info keys: {info.keys()}")
            if 'is_success' in info:
                print(f"Success metric: {info['is_success']}")
            
            # Run a few episodes
            print(f"\n--- Running 5 episodes ---")
            success_count = 0
            episode_lengths = []
            
            for ep in range(5):
                obs, info = env.reset()
                steps = 0
                episode_reward = 0
                
                for _ in range(50):  # Max 50 steps per episode
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    steps += 1
                    
                    if 'is_success' in info and info['is_success']:
                        success_count += 1
                        break
                    
                    if terminated or truncated:
                        break
                
                episode_lengths.append(steps)
                print(f"  Episode {ep + 1}: {steps} steps, reward={episode_reward:.2f}, success={info.get('is_success', False)}")
            
            print(f"\nSummary: {success_count}/5 successful episodes")
            print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
            
            env.close()
            return True
            
    except Exception as e:
        print(f"✗ Error testing Gymnasium-Robotics: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_goal_conditioned_compatibility():
    """Test compatibility with goal-conditioned RL algorithms."""
    print_section("GOAL-CONDITIONED RL COMPATIBILITY")
    
    print("Testing if environments can be adapted for HDM training...\n")
    
    # Test Gymnasium-Robotics (should work out of the box)
    print("--- Gymnasium-Robotics Compatibility ---")
    try:
        env = gym.make('FetchReach-v4')
        obs, _ = env.reset()
        
        if isinstance(obs, dict) and all(k in obs for k in ['observation', 'desired_goal', 'achieved_goal']):
            print("✓✓✓ Gymnasium-Robotics: FULLY COMPATIBLE!")
            print("    - Uses standard goal-conditioned format")
            print("    - Can be used directly with HDM/HER algorithms")
            print("    - Provides compute_reward for HER relabeling")
            
            # Test compute_reward function
            achieved = obs['achieved_goal']
            desired = obs['desired_goal']
            # Most gymnasium-robotics envs support compute_reward
            if hasattr(env, 'compute_reward'):
                reward = env.compute_reward(achieved, desired, {})
                print(f"    - compute_reward works: {reward}")
            
        env.close()
        
    except Exception as e:
        print(f"✗ Gymnasium-Robotics compatibility check failed: {e}")
    
    # Test Metaworld (needs wrapper)
    print("\n--- Metaworld Compatibility ---")
    try:
        import metaworld
        # Use MT1 (single task) instead of ML1
        mt1 = metaworld.MT1('reach-v3')  # Use v3 instead of v2
        env = mt1.train_classes['reach-v3']()
        task = random.sample(mt1.train_tasks, 1)[0]
        env.set_task(task)
        
        obs, info = env.reset()
        
        print("⚠ Metaworld: NEEDS WRAPPER")
        print("    - Returns flat observation vector")
        print("    - Needs wrapper to extract goal from observation")
        print("    - But environment has goal information embedded")
        print("    - Can be adapted with a simple wrapper")
        
        # Show that goal info is available in task
        if hasattr(task, 'data'):
            print(f"    - Task contains goal data: {list(task.data.keys())}")
        
    except Exception as e:
        print(f"⚠ Metaworld compatibility check issue: {e}")


def test_pytorch_compatibility():
    """Test PyTorch compatibility for neural network training."""
    print_section("PYTORCH COMPATIBILITY")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS (Metal) available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'N/A'}")
    
    # Test basic tensor operations
    print("\n--- Basic PyTorch Operations ---")
    x = torch.randn(32, 10)
    net = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4)
    )
    y = net(x)
    print(f"✓ Neural network forward pass successful: input {x.shape} -> output {y.shape}")
    
    # Test with environment observations
    print("\n--- Environment-to-PyTorch Pipeline ---")
    try:
        env = gym.make('FetchReach-v4')
        obs, _ = env.reset()
        
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(obs['observation'])
        goal_tensor = torch.FloatTensor(obs['desired_goal'])
        
        print(f"✓ Observation -> Tensor: {obs_tensor.shape}")
        print(f"✓ Goal -> Tensor: {goal_tensor.shape}")
        
        # Concatenate for policy input
        policy_input = torch.cat([obs_tensor, goal_tensor])
        print(f"✓ Policy input (obs + goal): {policy_input.shape}")
        
        env.close()
        
    except Exception as e:
        print(f"✗ Environment-PyTorch pipeline test failed: {e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("  TESTING GYMNASIUM-ROBOTICS AND METAWORLD FOR HDM PROJECT")
    print("=" * 80)
    
    results = {
        'metaworld': False,
        'gymnasium_robotics': False,
    }
    
    # Run tests
    results['metaworld'] = test_metaworld()
    results['gymnasium_robotics'] = test_gymnasium_robotics()
    
    test_goal_conditioned_compatibility()
    test_pytorch_compatibility()
    
    # Final summary
    print_section("FINAL SUMMARY & RECOMMENDATIONS")
    
    print("Environment Support:")
    print(f"  - Metaworld: {'✓ Working' if results['metaworld'] else '✗ Issues'}")
    print(f"  - Gymnasium-Robotics: {'✓ Working' if results['gymnasium_robotics'] else '✗ Issues'}")
    
    print("\n" + "-" * 80)
    print("RECOMMENDATIONS FOR HDM PROJECT:")
    print("-" * 80)
    
    print("\n1. GYMNASIUM-ROBOTICS (HIGHLY RECOMMENDED):")
    print("   ✓ Perfect goal-conditioned format out of the box")
    print("   ✓ Compatible with HDM/HER algorithms immediately")
    print("   ✓ Provides compute_reward for hindsight relabeling")
    print("   ✓ Well-maintained and documented")
    print("   ✓ Environments: FetchPush, FetchReach, HandManipulate, etc.")
    
    print("\n2. METAWORLD (USABLE WITH WRAPPER):")
    print("   ✓ Rich set of robotic manipulation tasks")
    print("   ✓ Push, Door, Drawer tasks similar to your current setup")
    print("   ⚠ Needs goal-conditioned wrapper (but straightforward)")
    print("   ✓ Multiple tasks per benchmark (good for meta-learning)")
    
    print("\n3. NEXT STEPS:")
    print("   1. Start with Gymnasium-Robotics FetchPush/FetchReach")
    print("   2. Adapt HDM code to use goal-conditioned dict observations")
    print("   3. Create wrapper for Metaworld environments if needed")
    print("   4. Compare performance with original custom environments")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

