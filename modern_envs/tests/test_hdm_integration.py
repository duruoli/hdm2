"""
Test HDM integration with modern environments.

This script mimics how HDM's __main__.py uses environments.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gym.spaces import Discrete

# This is how HDM imports environments
from modern_envs import create_env, get_env_params
from modern_envs.discretized_action_env import DiscretizedActionEnv


def test_hdm_workflow():
    """Test the complete HDM workflow with modern environment."""
    print("=" * 70)
    print("Testing HDM Integration with Modern SawyerPush")
    print("=" * 70)
    
    env_name = 'pusher'
    
    # Step 1: Create environment (like HDM does)
    print(f"\n1. Creating environment '{env_name}'...")
    env = create_env(env_name)
    env_params = get_env_params(env_name)
    print(f"   ✅ Created: {type(env).__name__}")
    print(f"   ✅ Params: {env_params}")
    
    # Step 2: Update params with observation/goal dimensions (like HDM does)
    print("\n2. Extracting observation/goal dimensions...")
    env_params.update({
        'obs_dim': env.observation_space.shape[0],
        'goal_dim': env.goal_space.shape[0],
        'independent_policy': False,
    })
    print(f"   ✅ Observation dim: {env_params['obs_dim']}")
    print(f"   ✅ Goal dim: {env_params['goal_dim']}")
    
    # Step 3: Discretize actions (like HDM does)
    print("\n3. Discretizing action space...")
    action_granularity = 3
    env_discretized = DiscretizedActionEnv(env, granularity=action_granularity)
    print(f"   ✅ Action space before: {env.action_space}")
    print(f"   ✅ Action space after: {env_discretized.action_space}")
    print(f"   ✅ Total discrete actions: {env_discretized.action_space.n}")
    assert isinstance(env_discretized.action_space, Discrete)
    
    # Step 4: Test environment interaction (like HDM does)
    print("\n4. Testing environment interaction...")
    state = env_discretized.reset()
    print(f"   - Initial state shape: {state.shape}")
    print(f"   - State contains: [obs({env_params['obs_dim']}), "
          f"goal({env_params['goal_dim']}), state_goal({env_params['goal_dim']})]")
    
    # Sample discrete action
    action = env_discretized.action_space.sample()
    print(f"   - Discrete action sampled: {action}")
    print(f"   - Maps to continuous: {env_discretized.base_actions[action]}")
    
    # Step
    next_state, reward, done, info = env_discretized.step(action)
    print(f"   - Next state shape: {next_state.shape}")
    print(f"   - Reward: {reward:.4f}")
    print(f"   ✅ Step successful")
    
    # Step 5: Test goal-conditioned methods (HDM uses these)
    print("\n5. Testing goal-conditioned methods...")
    
    # Extract observation and goal
    obs = env_discretized.observation(state)
    print(f"   - Observation shape: {obs.shape}")
    
    goal_repr = env_discretized.extract_goal(state)
    print(f"   - Goal representation shape: {goal_repr.shape}")
    
    # Sample new goal
    goal_state = env_discretized.sample_goal()
    print(f"   - Sampled goal state shape: {goal_state.shape}")
    
    # Compute distance
    distance = env_discretized.goal_distance(state, goal_state)
    print(f"   - Goal distance: {distance:.4f}")
    print(f"   ✅ All goal methods working")
    
    # Step 6: Run a short episode
    print("\n6. Running short episode...")
    state = env_discretized.reset()
    trajectory = [state]
    
    for step in range(10):
        action = env_discretized.action_space.sample()
        state, reward, done, info = env_discretized.step(action)
        trajectory.append(state)
    
    trajectory = np.array(trajectory)
    print(f"   - Episode length: {len(trajectory)}")
    print(f"   - Trajectory shape: {trajectory.shape}")
    print(f"   ✅ Episode completed")
    
    # Step 7: Test diagnostic logging
    print("\n7. Testing diagnostics...")
    goal_states = np.array([env_discretized.sample_goal() for _ in range(3)])
    trajectories = np.array([trajectory for _ in range(3)])
    
    diagnostics = env_discretized.get_diagnostics(trajectories, goal_states)
    print(f"   - Diagnostic keys: {list(diagnostics.keys())}")
    for key, value in list(diagnostics.items())[:3]:
        print(f"     • {key}: {value:.4f}")
    print(f"   ✅ Diagnostics working")
    
    print("\n" + "=" * 70)
    print("✅ ALL HDM INTEGRATION TESTS PASSED!")
    print("=" * 70)
    print("\nThe complete wrapper stack is working:")
    print("  MuJoCo (modern)")
    print("    └── SawyerPushAndReachXYEnvModern")
    print("        └── GymGoalEnvWrapper (flattens Dict obs)")
    print("            └── DiscretizedActionEnv (continuous → discrete)")
    print("                └── HDM Algorithm ✅")
    return True


if __name__ == '__main__':
    success = test_hdm_workflow()
    sys.exit(0 if success else 1)

