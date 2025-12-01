"""
Test that HDM training can start with modern environments.

This mimics what happens when you run: bash scripts/push.sh
"""

import sys
import os

# Make sure modern_envs can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Testing HDM Training Start with Modern SawyerPush")
print("=" * 70)

print("\n1. Checking modern_envs import...")
try:
    import modern_envs
    print("   ✅ modern_envs imported successfully")
    print(f"   ✅ Available: {modern_envs.ENV_NAMES}")
except ImportError as e:
    print(f"   ❌ Failed to import modern_envs: {e}")
    sys.exit(1)

print("\n2. Simulating HDM import (from hdm/__main__.py)...")
try:
    # This is what hdm/__main__.py does now
    import modern_envs as envs
    from modern_envs.discretized_action_env import DiscretizedActionEnv
    print("   ✅ HDM will use modern environments")
except ImportError:
    print("   ⚠️  Would fall back to gcsl (old mujoco_py)")

print("\n3. Testing environment creation (what HDM launch() does)...")
try:
    env_name = 'pusher'
    env = envs.create_env(env_name)
    env_params = envs.get_env_params(env_name)
    print(f"   ✅ Created {env_name} environment")
    print(f"   ✅ Params: {env_params}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Testing environment setup (what HDM does)...")
try:
    env_params.update({
        'obs_dim': env.observation_space.shape[0],
        'goal_dim': env.goal_space.shape[0],
        'independent_policy': False
    })
    print(f"   ✅ obs_dim: {env_params['obs_dim']}")
    print(f"   ✅ goal_dim: {env_params['goal_dim']}")
    
    # Discretize (what HDM does)
    from gym.spaces import Discrete
    if not isinstance(env.action_space, Discrete):
        granularity = 3
        env = DiscretizedActionEnv(env, granularity=granularity)
        print(f"   ✅ Discretized to {env.action_space.n} actions")
    
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n5. Testing basic episode (sanity check)...")
try:
    state = env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
    print(f"   ✅ Ran 5 steps successfully")
    print(f"   ✅ Final reward: {reward:.4f}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ HDM TRAINING SETUP SUCCESSFUL!")
print("=" * 70)
print("\nYou can now run: bash scripts/push.sh")
print("HDM will automatically use modern MuJoCo environments (no compilation!)")
print("=" * 70)


