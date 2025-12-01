"""
Metaworld environment integration for HDM.

Supports Metaworld's goal-conditioned robotic manipulation tasks.
Install: pip install metaworld

Note: Requires Metaworld 0.4+ (V3 environments).
If you have an older version with V2 environments, upgrade with:
    pip install --upgrade metaworld
"""

import numpy as np
from modern_envs.wrappers.metaworld_wrapper import MetaworldGoalWrapper


def create_metaworld_env(env_name, render_mode=None):
    """
    Create a Metaworld environment wrapped for HDM.
    
    Args:
        env_name: Metaworld task name (e.g., 'push-v3', 'door-open-v3')
        render_mode: Render mode ('human', 'rgb_array', or None)
    
    Returns:
        Wrapped environment compatible with HDM
    """
    try:
        import metaworld
    except ImportError:
        raise ImportError(
            "Metaworld not installed. Install with: pip install metaworld"
        )
    
    # Create benchmark to access environments
    ml1 = metaworld.ML1(env_name)
    
    # Get environment class and create instance
    env_cls = ml1.train_classes[env_name]
    env = env_cls(render_mode=render_mode)
    
    # Set a task (for ML1, there's typically one task per environment)
    task = ml1.train_tasks[0]
    env.set_task(task)
    
    # Wrap for HDM
    # Metaworld goals are typically 3D positions (xyz)
    wrapped_env = MetaworldGoalWrapper(env, goal_dim=3)
    
    return wrapped_env


def get_metaworld_env_params(env_name):
    """
    Get environment parameters for Metaworld environments.
    
    Args:
        env_name: Metaworld task name
    
    Returns:
        Dictionary of environment parameters for HDM
    """
    # Common parameters for Metaworld environments
    params = {
        'goal_threshold': 0.05,  # Standard threshold for success
        'max_trajectory_length': 500,  # Metaworld default episode length
        'max_timesteps': 1e6,
        'eval_freq': 10000,
        'eval_episodes': 50,
    }
    
    # Environment-specific adjustments
    env_specific = {
        'push-v3': {
            'goal_threshold': 0.05,
            'max_trajectory_length': 500,
        },
        'door-open-v3': {
            'goal_threshold': 0.08,
            'max_trajectory_length': 500,
        },
        'door-close-v3': {
            'goal_threshold': 0.08,
            'max_trajectory_length': 500,
        },
        'reach-v3': {
            'goal_threshold': 0.05,
            'max_trajectory_length': 500,
        },
        'pick-place-v3': {
            'goal_threshold': 0.05,
            'max_trajectory_length': 500,
        },
    }
    
    if env_name in env_specific:
        params.update(env_specific[env_name])
    
    return params


# Mapping from simplified names to Metaworld task names
# Note: Metaworld 0.4+ uses V3 environments
METAWORLD_ENV_MAP = {
    'metaworld_push': 'push-v3',
    'metaworld_door_open': 'door-open-v3',
    'metaworld_door_close': 'door-close-v3',
    'metaworld_reach': 'reach-v3',
    'metaworld_pick_place': 'pick-place-v3',
    'metaworld_button_press': 'button-press-v3',
    'metaworld_drawer_open': 'drawer-open-v3',
    'metaworld_drawer_close': 'drawer-close-v3',
    'metaworld_window_open': 'window-open-v3',
    'metaworld_window_close': 'window-close-v3',
}


def create_metaworld_env_by_name(simple_name, render_mode=None):
    """
    Create Metaworld environment using simplified name.
    
    Args:
        simple_name: Simplified name (e.g., 'metaworld_push')
        render_mode: Render mode
    
    Returns:
        Wrapped environment
    """
    if simple_name not in METAWORLD_ENV_MAP:
        raise ValueError(
            f"Unknown Metaworld environment: {simple_name}. "
            f"Available: {list(METAWORLD_ENV_MAP.keys())}"
        )
    
    metaworld_name = METAWORLD_ENV_MAP[simple_name]
    return create_metaworld_env(metaworld_name, render_mode)


def get_metaworld_env_params_by_name(simple_name):
    """Get parameters using simplified name."""
    if simple_name not in METAWORLD_ENV_MAP:
        raise ValueError(
            f"Unknown Metaworld environment: {simple_name}. "
            f"Available: {list(METAWORLD_ENV_MAP.keys())}"
        )
    
    metaworld_name = METAWORLD_ENV_MAP[simple_name]
    return get_metaworld_env_params(metaworld_name)

