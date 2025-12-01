# Using Metaworld and Gymnasium-Robotics Environments with HDM

This guide shows how to use pre-packaged goal-conditioned environments from Metaworld and Gymnasium-Robotics with the HDM algorithm.

## Installation

### Install Metaworld
```bash
pip install metaworld

# Note: This integration requires Metaworld 0.4+ (V3 environments)
# If you have an older version, upgrade with: pip install --upgrade metaworld
```

### Install Gymnasium-Robotics
```bash
pip install gymnasium-robotics
```

## Available Environments

### Metaworld Environments

Metaworld provides 50 diverse robotic manipulation tasks. Here are some commonly used ones:

- `metaworld_push` - Push a puck to a goal position
- `metaworld_reach` - Reach to a goal position with the gripper
- `metaworld_door_open` - Open a door
- `metaworld_door_close` - Close a door
- `metaworld_pick_place` - Pick up an object and place it at a goal
- `metaworld_button_press` - Press a button
- `metaworld_drawer_open` - Open a drawer
- `metaworld_drawer_close` - Close a drawer
- `metaworld_window_open` - Open a window
- `metaworld_window_close` - Close a window

### Gymnasium-Robotics Environments

#### Fetch Robot (Arm manipulation)
- `fetch_reach` - Reach to a goal position
- `fetch_push` - Push a block to a goal position
- `fetch_slide` - Slide a puck to a goal position
- `fetch_pick_place` - Pick and place a block

#### Shadow Dexterous Hand (Dexterous manipulation)
- `hand_reach` - Reach with fingertips
- `hand_manipulate_block` - Manipulate a block
- `hand_manipulate_egg` - Manipulate an egg-shaped object
- `hand_manipulate_pen` - Manipulate a pen (rotate to target orientation)

#### Point Maze (Navigation)
- `pointmaze_open` - Navigate in an open space
- `pointmaze_umaze` - Navigate through a U-shaped maze
- `pointmaze_medium` - Navigate through a medium-sized maze
- `pointmaze_large` - Navigate through a large maze

## Basic Usage

### Quick Test

Test that environments are working correctly:

```bash
# Test all available environments (interface only)
python test_external_envs_integration.py

# Test a specific environment
python test_external_envs_integration.py --env metaworld_push

# Run a quick training test (2 epochs)
python test_external_envs_integration.py --env metaworld_push --train

# Run longer training test
python test_external_envs_integration.py --env hand_manipulate_pen --train --epochs 5
```

### Training with HDM

Use the same command-line interface as before, just specify the new environment name:

```bash
# Train on Metaworld push task
python -m hdm --env_name metaworld_push \
    --n_cycles 40 \
    --n_initial_rollouts 200 \
    --n_test_rollouts 50 \
    --independent_policy

# Train on Gymnasium-Robotics hand manipulation task
python -m hdm --env_name hand_manipulate_pen \
    --n_cycles 40 \
    --n_initial_rollouts 200 \
    --n_test_rollouts 50 \
    --independent_policy

# Train on Point Maze task
python -m hdm --env_name pointmaze_medium \
    --n_cycles 40 \
    --n_initial_rollouts 200 \
    --n_test_rollouts 50 \
    --independent_policy
```

## Programmatic Usage

### Creating Environments in Code

```python
import modern_envs as envs

# Method 1: Use the unified interface
env = envs.create_env('metaworld_push')
env_params = envs.get_env_params('metaworld_push')

# Method 2: Use specific factory functions
from modern_envs.envs import create_metaworld_env_by_name
env = create_metaworld_env_by_name('metaworld_push')

# Method 3: Direct Metaworld access
from modern_envs.envs import create_metaworld_env
env = create_metaworld_env('push-v3')  # Uses Metaworld's original naming

# Method 4: Gymnasium-Robotics
from modern_envs.envs import create_gymnasium_robotics_env_by_name
env = create_gymnasium_robotics_env_by_name('hand_manipulate_pen')
```

### Environment Interface

All environments follow the HDM interface:

```python
# Reset returns state (obs + goal concatenated)
state = env.reset()  # shape: (obs_dim + goal_dim,)

# Step returns 4 values (old gym API)
state, reward, done, info = env.step(action)

# Extract observation and goal from state
obs = env.observation(state)      # shape: (obs_dim,)
goal = env.extract_goal(state)    # shape: (goal_dim,)

# Sample a random goal state for HER
goal_state = env.sample_goal()    # shape: (obs_dim + goal_dim,)

# Compute distance between states' goals
distance = env.goal_distance(state1, state2)

# Seed for reproducibility
env.seed(42)
```

## Environment Parameters

Each environment comes with sensible default parameters:

### Metaworld
- `goal_threshold`: 0.05 - 0.08 (task-dependent)
- `max_trajectory_length`: 500
- `max_timesteps`: 1e6
- `eval_freq`: 10000

### Fetch Environments
- `goal_threshold`: 0.05
- `max_trajectory_length`: 50
- `max_timesteps`: 1e6
- `eval_freq`: 10000

### Hand Environments
- `goal_threshold`: 0.01 (requires high precision)
- `max_trajectory_length`: 50
- `max_timesteps`: 1e6
- `eval_freq`: 10000

### Point Maze Environments
- `goal_threshold`: 0.45
- `max_trajectory_length`: 300-600 (size-dependent)
- `max_timesteps`: 1e6
- `eval_freq`: 10000

## Tips for Training

### Action Discretization

All continuous action spaces are automatically discretized. For large action spaces (>100 discrete actions), use `--independent_policy`:

```bash
python -m hdm --env_name metaworld_push --independent_policy
```

### Hyperparameters

Good starting hyperparameters:

```bash
python -m hdm --env_name metaworld_push \
    --independent_policy \
    --n_cycles 40 \
    --optimize_every 50 \
    --n_batches 50 \
    --batch_size 256 \
    --buffer_size 1000000 \
    --future_p 0.85 \
    --lr_actor 5e-4 \
    --n_initial_rollouts 200 \
    --n_test_rollouts 50 \
    --hdm_q_coef 1.0 \
    --hdm_bc \
    --hdm_weights_to_indicator
```

### For Complex Tasks

For more difficult tasks (e.g., hand manipulation, large mazes):
- Increase `n_initial_rollouts` to 500-1000
- Increase `n_cycles` to 100+
- Adjust `goal_threshold` if needed
- Consider increasing `max_trajectory_length`

## Architecture Overview

```
External Environment (Metaworld/Gymnasium-Robotics)
                ↓
    DictGoalEnvWrapper (converts dict obs to state)
                ↓
    DiscretizedActionEnv (discretizes continuous actions)
                ↓
            HDM Algorithm
```

The `DictGoalEnvWrapper` converts from dict observations:
```python
obs = {
    'observation': array(...),
    'achieved_goal': array(...),
    'desired_goal': array(...)
}
```

To HDM state format:
```python
state = concatenate([observation, desired_goal])
```

## Troubleshooting

### Environment not found
```
ValueError: Unknown environment: metaworld_push
```
**Solution**: Install the required package (`pip install metaworld` or `pip install gymnasium-robotics`)

### Import errors
```
ImportError: metaworld not installed
```
**Solution**: The integration gracefully handles missing packages. Install the package to use those environments.

### Shape mismatches
If you see shape errors, verify that:
1. The environment wrapper is correctly converting dict observations to states
2. The state space is correctly defined as `obs_dim + goal_dim`
3. `reset()` and `step()` return states, not observations

## Adding New Environments

To add more environments from these packages:

1. **Metaworld**: Edit `modern_envs/envs/metaworld_envs.py` and add to `METAWORLD_ENV_MAP`
2. **Gymnasium-Robotics**: Edit `modern_envs/envs/gymnasium_robotics_envs.py` and add to `GYMNASIUM_ROBOTICS_ENV_MAP`

Example:
```python
# In metaworld_envs.py
METAWORLD_ENV_MAP = {
    'metaworld_push': 'push-v2',
    'metaworld_my_new_task': 'my-new-task-v2',  # Add this line
}
```

Then use it:
```bash
python -m hdm --env_name metaworld_my_new_task --independent_policy
```

## Complete Example

Here's a complete example script:

```python
#!/usr/bin/env python
"""Train HDM on a Metaworld task."""

import argparse
import os
import modern_envs as envs
from hdm import launch

def main():
    parser = argparse.ArgumentParser()
    
    # Environment
    parser.add_argument('--env_name', type=str, default='metaworld_push')
    parser.add_argument('--seed', type=int, default=0)
    
    # Training
    parser.add_argument('--n_cycles', type=int, default=40)
    parser.add_argument('--n_initial_rollouts', type=int, default=200)
    
    # HDM
    parser.add_argument('--independent_policy', action='store_true')
    parser.add_argument('--hdm_q_coef', type=float, default=1.0)
    
    # ... (add other args as needed)
    
    args = parser.parse_args()
    
    # Launch training
    algo = launch(args)
    algo.run()

if __name__ == '__main__':
    main()
```

## References

- **Metaworld**: https://github.com/Farama-Foundation/Metaworld
- **Gymnasium-Robotics**: https://robotics.farama.org/
- **HDM Paper**: Understanding Hindsight Goal Relabeling Requires Rethinking Divergence Minimization

