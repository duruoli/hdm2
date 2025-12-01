"""
Gymnasium-Robotics environment integration for HDM.

Supports goal-conditioned environments from Gymnasium-Robotics:
- Fetch robot arm (stable): FetchReach, FetchPush, FetchSlide, FetchPickAndPlace
- Shadow Dexterous Hand (HandManipulate* series): HandManipulateBlock, Egg, Pen, etc.
- Adroit Hand (older system, may have MuJoCo issues): AdroitHandPen, AdroitHandDoor, etc.
- Point Maze (navigation, stable): PointMaze_Open, PointMaze_Medium, etc.
- Ant Maze (quadruped navigation): AntMaze_Medium, AntMaze_Large

Install: pip install gymnasium-robotics

Note: 
- Shadow Hand (HandManipulate*) is the modern dexterous hand system
- Adroit Hand may have MuJoCo XML compatibility issues with some versions
- Use Fetch, Shadow Hand, or PointMaze for most stable experience
"""

import numpy as np
from modern_envs.wrappers.dict_goal_env_wrapper import DictGoalEnvWrapper


def create_gymnasium_robotics_env(env_id, render_mode=None, **kwargs):
    """
    Create a Gymnasium-Robotics environment wrapped for HDM.
    
    Args:
        env_id: Gymnasium environment ID (e.g., 'FetchPush-v2', 'HandReach-v1')
        render_mode: Render mode ('human', 'rgb_array', or None)
        **kwargs: Additional arguments for the environment
    
    Returns:
        Wrapped environment compatible with HDM
    """
    try:
        import gymnasium as gym
        import gymnasium_robotics  # This registers the robotics environments
    except ImportError:
        raise ImportError(
            "Gymnasium not installed. Install with: pip install gymnasium gymnasium-robotics"
        )
    
    # Create the base environment
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    
    # Check if environment has Dict observation space (goal-conditioned)
    try:
        from gymnasium.spaces import Dict as GymnasiumDict
    except ImportError:
        GymnasiumDict = None
    
    try:
        from gym.spaces import Dict as GymDict
    except ImportError:
        GymDict = None
    
    is_dict_space = False
    if GymnasiumDict is not None and isinstance(env.observation_space, GymnasiumDict):
        is_dict_space = True
    if GymDict is not None and isinstance(env.observation_space, GymDict):
        is_dict_space = True
    
    # Wrap based on observation space type
    if is_dict_space:
        # Standard goal-conditioned environments (Fetch, PointMaze, Shadow Hand)
        wrapped_env = DictGoalEnvWrapper(env, goal_key='desired_goal', achieved_goal_key='achieved_goal')
        return wrapped_env
    elif 'Adroit' in env_id:
        # Adroit Hand environments have Box observations with embedded goals
        from modern_envs.wrappers.box_goal_env_wrapper import BoxGoalEnvWrapper
        # Use door handle position (randomized) as goal, palm position as achieved goal
        wrapped_env = BoxGoalEnvWrapper(
            env,
            goal_indices=[32, 33, 34],  # Door handle xyz position (randomized each reset)
            achieved_goal_indices=[29, 30, 31],  # Palm xyz position
            obs_indices=list(range(29))  # Joint angles and other state
        )
        return wrapped_env
    else:
        # Other Box observation environments are not supported
        raise ValueError(
            f"Environment '{env_id}' has Box observation space without goal structure. "
            f"Compatible environments: Fetch*, PointMaze*, HandManipulate* (Shadow Hand), Adroit*"
        )


def get_gymnasium_robotics_env_params(env_id):
    """
    Get environment parameters for Gymnasium-Robotics environments.
    
    Args:
        env_id: Gymnasium environment ID
    
    Returns:
        Dictionary of environment parameters for HDM
    """
    # Common parameters
    params = {
        'goal_threshold': 0.05,
        'max_timesteps': 1e6,
        'eval_freq': 10000,
        'eval_episodes': 50,
    }
    
    # Environment-specific parameters
    if 'Fetch' in env_id:
        params.update({
            'goal_threshold': 0.05,
            'max_trajectory_length': 50,
        })
    elif 'Hand' in env_id:
        params.update({
            'goal_threshold': 0.01,  # Hand tasks need higher precision
            'max_trajectory_length': 50,
        })
    elif 'PointMaze' in env_id:
        params.update({
            'goal_threshold': 0.45,  # Maze tasks have larger threshold
            'max_trajectory_length': 300 if 'Large' in env_id else 600,
        })
    else:
        # Default
        params['max_trajectory_length'] = 50
    
    return params


# Mapping from simplified names to Gymnasium-Robotics environment IDs
GYMNASIUM_ROBOTICS_ENV_MAP = {
    # Fetch robot arm environments (v4 is latest, stable)
    'fetch_reach': 'FetchReach-v4',
    'fetch_push': 'FetchPush-v4',
    'fetch_slide': 'FetchSlide-v4',
    'fetch_pick_place': 'FetchPickAndPlace-v4',
    
    # Adroit Hand environments (Box observation space with embedded goals)
    'adroit_hand_door': 'AdroitHandDoor-v1',
    'adroit_hand_hammer': 'AdroitHandHammer-v1',
    'adroit_hand_pen': 'AdroitHandPen-v1',
    'adroit_hand_relocate': 'AdroitHandRelocate-v1',
    
    # Backward compatibility aliases for Adroit Hand
    'hand_door': 'AdroitHandDoor-v1',
    'hand_hammer': 'AdroitHandHammer-v1',
    'hand_pen': 'AdroitHandPen-v1',
    'hand_relocate': 'AdroitHandRelocate-v1',
    
    # Shadow Dexterous Hand environments (HandManipulate* series)
    # These use the Shadow Dexterous Hand robot (more modern than Adroit)
    'shadow_hand_block': 'HandManipulateBlock-v1',
    'shadow_hand_block_full': 'HandManipulateBlockFull-v1',
    'shadow_hand_block_rotate': 'HandManipulateBlockRotateParallel-v1',
    'shadow_hand_egg': 'HandManipulateEgg-v1',
    'shadow_hand_egg_full': 'HandManipulateEggFull-v1',
    'shadow_hand_egg_rotate': 'HandManipulateEggRotateParallel-v1',
    'shadow_hand_pen': 'HandManipulatePen-v1',
    'shadow_hand_pen_full': 'HandManipulatePenFull-v1',
    'shadow_hand_pen_rotate': 'HandManipulatePenRotateParallel-v1',
    
    # Point Maze environments (navigation, stable)
    'pointmaze_open': 'PointMaze_Open-v3',
    'pointmaze_umaze': 'PointMaze_UMaze-v3',
    'pointmaze_medium': 'PointMaze_Medium-v3',
    'pointmaze_large': 'PointMaze_Large-v3',
    
    # Ant Maze environments (quadruped navigation)
    'antmaze_medium': 'AntMaze_Medium-v3',
    'antmaze_large': 'AntMaze_Large-v3',
}


def create_gymnasium_robotics_env_by_name(simple_name, render_mode=None, **kwargs):
    """
    Create Gymnasium-Robotics environment using simplified name.
    
    Args:
        simple_name: Simplified name (e.g., 'hand_manipulate_pen')
        render_mode: Render mode
        **kwargs: Additional environment arguments
    
    Returns:
        Wrapped environment
    """
    if simple_name not in GYMNASIUM_ROBOTICS_ENV_MAP:
        raise ValueError(
            f"Unknown Gymnasium-Robotics environment: {simple_name}. "
            f"Available: {list(GYMNASIUM_ROBOTICS_ENV_MAP.keys())}"
        )
    
    env_id = GYMNASIUM_ROBOTICS_ENV_MAP[simple_name]
    return create_gymnasium_robotics_env(env_id, render_mode, **kwargs)


def get_gymnasium_robotics_env_params_by_name(simple_name):
    """Get parameters using simplified name."""
    if simple_name not in GYMNASIUM_ROBOTICS_ENV_MAP:
        raise ValueError(
            f"Unknown Gymnasium-Robotics environment: {simple_name}. "
            f"Available: {list(GYMNASIUM_ROBOTICS_ENV_MAP.keys())}"
        )
    
    env_id = GYMNASIUM_ROBOTICS_ENV_MAP[simple_name]
    return get_gymnasium_robotics_env_params(env_id)

