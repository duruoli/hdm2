"""
Test script for modern SawyerPush environment.

This verifies that the modern implementation works correctly.
"""

import sys
import os

# Add project root and dependencies to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'gcsl', 'dependencies'))

import numpy as np
from modern_envs.sawyer_push_modern import SawyerPushGoalEnv


def test_sawyer_push():
    """Test basic functionality of SawyerPush environment."""
    print("=" * 60)
    print("Testing Modern SawyerPush Environment")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating environment...")
    try:
        env = SawyerPushGoalEnv(fixed_start=True, fixed_goal=False)
        print("   ✅ Environment created successfully")
    except Exception as e:
        print(f"   ❌ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check spaces
    print("\n2. Checking observation/action spaces...")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Goal space: {env.goal_space}")
    print(f"   - Action space: {env.action_space}")
    print("   ✅ Spaces defined")
    
    # Test reset
    print("\n3. Testing reset...")
    try:
        state = env.reset()
        print(f"   - State shape: {state.shape}")
        print(f"   - State sample: {state[:4]}...")
        print("   ✅ Reset successful")
    except Exception as e:
        print(f"   ❌ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test step
    print("\n4. Testing environment step...")
    try:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(f"   - Action: {action}")
        print(f"   - Next state shape: {next_state.shape}")
        print(f"   - Reward: {reward}")
        print(f"   - Done: {done}")
        print("   ✅ Step successful")
    except Exception as e:
        print(f"   ❌ Step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test multiple steps
    print("\n5. Testing multiple steps...")
    try:
        state = env.reset()
        for i in range(10):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
        print(f"   - Ran 10 steps successfully")
        print(f"   - Final state sample: {state[:4]}...")
        print("   ✅ Multiple steps successful")
    except Exception as e:
        print(f"   ❌ Multiple steps failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test goal methods
    print("\n6. Testing goal-conditioned methods...")
    try:
        goal = env.sample_goal()
        print(f"   - Sampled goal shape: {goal.shape}")
        
        obs = env.observation(state)
        print(f"   - Observation shape: {obs.shape}")
        
        goal_repr = env.extract_goal(state)
        print(f"   - Extracted goal shape: {goal_repr.shape}")
        
        distance = env.goal_distance(state, goal)
        print(f"   - Goal distance: {distance}")
        print("   ✅ Goal methods working")
    except Exception as e:
        print(f"   ❌ Goal methods failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nSawyerPush environment is working with modern MuJoCo!")
    return True


if __name__ == '__main__':
    success = test_sawyer_push()
    sys.exit(0 if success else 1)

