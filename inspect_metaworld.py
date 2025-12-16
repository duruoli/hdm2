"""Quick script to inspect MetaWorld push-v3 observation space"""
import metaworld
import numpy as np

# Create push-v3 environment
ml1 = metaworld.ML1('door-open-v3')
env_cls = ml1.train_classes['door-open-v3']
env = env_cls()
task = ml1.train_tasks[0]
env.set_task(task)

# Check observation space
print("=" * 60)
print("MetaWorld push-v3 Observation Space")
print("=" * 60)
print(f"\nObservation Space Type: {type(env.observation_space)}")
print(f"Observation Space: {env.observation_space}")
print(f"Shape: {env.observation_space.shape}")

# Reset and get an actual observation
obs, info = env.reset()
print(f"\n\nActual Observation Shape: {obs.shape}")
print(f"Observation values:\n{obs}")

# Try to get more info from the environment
print(f"\n\nEnvironment attributes:")
if hasattr(env, '_get_obs'):
    print("Has _get_obs method")
if hasattr(env, 'observation_space'):
    print(f"observation_space.low[:10]: {env.observation_space.low[:10]}")
    print(f"observation_space.high[:10]: {env.observation_space.high[:10]}")

# Check what info dict contains
print(f"\n\nInfo dict keys: {info.keys() if info else 'No info'}")
if info and 'success' in info:
    print(f"Has 'success' in info: {info['success']}")

# Try to understand observation structure by checking env source
print(f"\n\nTrying to understand observation structure:")
print(f"Total observation dim: {obs.shape[0]}")

# Reset a few times and see what changes
obs1, _ = env.reset()
obs2, _ = env.reset()
print(f"\n\nComparing two resets:")
print(f"obs1[-3:] (likely goal): {obs1[-3:]}")
print(f"obs2[-3:] (likely goal): {obs2[-3:]}")
print(f"obs1[0:3] (likely gripper): {obs1[0:3]}")
print(f"obs2[0:3] (likely gripper): {obs2[0:3]}")
print(f"obs1[4:7] (likely object): {obs1[4:7]}")
print(f"obs2[4:7] (likely object): {obs2[4:7]}")

# Take a step and see what happens
action = env.action_space.sample()
obs_next, reward, terminated, truncated, info = env.step(action)
print(f"\n\nAfter one step:")
print(f"Gripper pos change: {np.linalg.norm(obs_next[0:3] - obs1[0:3]):.4f}")
print(f"Object pos change: {np.linalg.norm(obs_next[4:7] - obs1[4:7]):.4f}")
print(f"Goal change: {np.linalg.norm(obs_next[-3:] - obs1[-3:]):.4f}")
print(f"Reward: {reward}")

# Check for internal goal/target position
print(f"\n\nSearching for goal/target in environment:")
for attr in dir(env):
    if 'goal' in attr.lower() or 'target' in attr.lower():
        try:
            val = getattr(env, attr)
            if not callable(val):
                print(f"  {attr}: {val}")
        except:
            pass

# Common attributes to check
print(f"\n\nChecking common goal attributes:")
if hasattr(env, '_target_pos'):
    print(f"env._target_pos: {env._target_pos}")
if hasattr(env, 'goal'):
    print(f"env.goal: {env.goal}")
if hasattr(env, '_goal'):
    print(f"env._goal: {env._goal}")
if hasattr(env, 'goal_space'):
    print(f"env.goal_space: {env.goal_space}")

# Check distance to goal from current object position
print(f"\n\nTrying to find goal by checking task object:")
print(f"Task object: {task}")
if hasattr(task, 'data'):
    print(f"Task data keys: {task.data.keys() if hasattr(task.data, 'keys') else task.data}")
