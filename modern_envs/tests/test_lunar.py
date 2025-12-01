"""
Test Lunar Lander environment with new structure.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

print("=" * 70)
print("Testing Lunar Lander Environment (Reorganized Structure)")
print("=" * 70)

# Test imports with new structure
print("\n1. Testing imports...")
try:
    from modern_envs import create_env, get_env_params
    from modern_envs.envs import LunarEnv
    from modern_envs.wrappers import DiscretizedActionEnv
    print("   ✅ All imports successful with new structure")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test creating Lunar environment
print("\n2. Creating Lunar environment...")
try:
    env = create_env('lunar')
    env_params = get_env_params('lunar')
    print(f"   ✅ Created: {type(env).__name__}")
    print(f"   ✅ Params: {env_params}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test environment spaces
print("\n3. Checking environment spaces...")
print(f"   - Observation space: {env.observation_space}")
print(f"   - Goal space: {env.goal_space}")
print(f"   - Action space: {env.action_space}")
print(f"   ✅ Spaces defined")

# Test reset
print("\n4. Testing reset...")
try:
    state = env.reset()
    print(f"   - State shape: {state.shape}")
    print(f"   - State sample: {state[:4]}...")
    print("   ✅ Reset successful")
except Exception as e:
    print(f"   ❌ Reset failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test step
print("\n5. Testing environment step...")
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
    sys.exit(1)

# Test goal-conditioned methods
print("\n6. Testing goal-conditioned methods...")
try:
    # Sample goal
    goal = env.sample_goal()
    print(f"   - Sampled goal shape: {goal.shape}")
    
    # Extract observation
    obs = env.observation(state)
    print(f"   - Observation shape: {obs.shape}")
    
    # Extract goal
    goal_repr = env.extract_goal(state)
    print(f"   - Extracted goal shape: {goal_repr.shape}")
    
    # Compute distance
    distance = env.goal_distance(state, goal)
    print(f"   - Goal distance: {distance}")
    print("   ✅ Goal methods working")
except Exception as e:
    print(f"   ❌ Goal methods failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test discretization (for HDM)
print("\n7. Testing action discretization...")
try:
    env_params.update({
        'obs_dim': env.observation_space.shape[0],
        'goal_dim': env.goal_space.shape[0],
    })
    
    env_discrete = DiscretizedActionEnv(env, granularity=3)
    print(f"   - Original action space: {env.action_space}")
    print(f"   - Discretized action space: {env_discrete.action_space}")
    print(f"   - Total discrete actions: {env_discrete.action_space.n}")
    
    # Test discretized step
    action = env_discrete.action_space.sample()
    state, reward, done, info = env_discrete.step(action)
    if env_discrete.base_actions is not None:
        print(f"   - Discrete action {action} → continuous {env_discrete.base_actions[action]}")
    else:
        print(f"   - Already discrete action: {action}")
    print("   ✅ Discretization working (pass-through for already discrete)")
except Exception as e:
    print(f"   ❌ Discretization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL LUNAR LANDER TESTS PASSED!")
print("=" * 70)
print("\nLunar Lander is ready for HDM training!")
print("Run: python -m hdm --env_name lunar ...")

