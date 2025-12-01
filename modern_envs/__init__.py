"""
Modern MuJoCo environments for HDM/GCSL using mujoco (not mujoco_py).

This module provides goal-conditioned robotic environments that work with:
- Modern MuJoCo (>= 2.2.0) instead of legacy mujoco_py
- Existing HDM algorithm
- No dependencies on old gcsl/multiworld packages (for MuJoCo envs)
"""

# Import core classes
from modern_envs.core import ModernMujocoEnv, GoalEnv, GymGoalEnvWrapper
from modern_envs.wrappers import DiscretizedActionEnv, Discretized
from modern_envs.utils import Serializable, MultitaskEnv

# Import specific environments
from modern_envs.envs.sawyer_push import SawyerPushGoalEnv
from modern_envs.envs.lunar_lander import LunarEnv

# Try to import external environment integrations (optional)
try:
    from modern_envs.envs.metaworld_envs import (
        create_metaworld_env_by_name,
        get_metaworld_env_params_by_name,
        METAWORLD_ENV_MAP,
    )
    METAWORLD_AVAILABLE = True
except ImportError:
    METAWORLD_AVAILABLE = False
    METAWORLD_ENV_MAP = {}

try:
    from modern_envs.envs.gymnasium_robotics_envs import (
        create_gymnasium_robotics_env_by_name,
        get_gymnasium_robotics_env_params_by_name,
        GYMNASIUM_ROBOTICS_ENV_MAP,
    )
    GYMNASIUM_ROBOTICS_AVAILABLE = True
except ImportError:
    GYMNASIUM_ROBOTICS_AVAILABLE = False
    GYMNASIUM_ROBOTICS_ENV_MAP = {}


# Environment registry - includes native and external environments
ENV_NAMES = ['pusher', 'lunar', 'pointmass_rooms', 'pointmass_empty']

# Add external environments if available
if METAWORLD_AVAILABLE:
    ENV_NAMES.extend(list(METAWORLD_ENV_MAP.keys()))

if GYMNASIUM_ROBOTICS_AVAILABLE:
    ENV_NAMES.extend(list(GYMNASIUM_ROBOTICS_ENV_MAP.keys()))


def create_env(env_name):
    """
    Create an environment by name (matching HDM's API).
    
    Args:
        env_name: One of ENV_NAMES
        
    Returns:
        Environment instance (not yet discretized - HDM does that)
    """
    # Native environments
    if env_name == 'pusher':
        return SawyerPushGoalEnv(fixed_start=True, fixed_goal=False)
    elif env_name == 'lunar':
        return LunarEnv(fixed_start=True, fixed_goal=False)
    elif env_name == 'pointmass_rooms':
        # TODO: Port pointmass when needed
        raise NotImplementedError(f"{env_name} not yet ported to modern MuJoCo")
    elif env_name == 'pointmass_empty':
        # TODO: Port pointmass when needed
        raise NotImplementedError(f"{env_name} not yet ported to modern MuJoCo")
    
    # Metaworld environments
    elif METAWORLD_AVAILABLE and env_name in METAWORLD_ENV_MAP:
        return create_metaworld_env_by_name(env_name)
    
    # Gymnasium-Robotics environments
    elif GYMNASIUM_ROBOTICS_AVAILABLE and env_name in GYMNASIUM_ROBOTICS_ENV_MAP:
        return create_gymnasium_robotics_env_by_name(env_name)
    
    else:
        available = ['pusher', 'lunar', 'pointmass_rooms', 'pointmass_empty']
        if METAWORLD_AVAILABLE:
            available.extend(list(METAWORLD_ENV_MAP.keys()))
        if GYMNASIUM_ROBOTICS_AVAILABLE:
            available.extend(list(GYMNASIUM_ROBOTICS_ENV_MAP.keys()))
        raise ValueError(f"Unknown environment: {env_name}. Available: {available}")


def get_env_params(env_name, images=False):
    """
    Get environment parameters (matching HDM's API).
    
    Args:
        env_name: One of ENV_NAMES
        images: Whether to use image observations
        
    Returns:
        Dictionary of environment parameters
    """
    # Metaworld environments
    if METAWORLD_AVAILABLE and env_name in METAWORLD_ENV_MAP:
        return get_metaworld_env_params_by_name(env_name)
    
    # Gymnasium-Robotics environments
    if GYMNASIUM_ROBOTICS_AVAILABLE and env_name in GYMNASIUM_ROBOTICS_ENV_MAP:
        return get_gymnasium_robotics_env_params_by_name(env_name)
    
    # Native environments
    base_params = dict(
        eval_freq=10000,
        eval_episodes=50,
        max_trajectory_length=50,
        max_timesteps=1e6,
    )

    if env_name == 'pusher':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif 'pointmass' in env_name:
        env_specific_params = dict(
            goal_threshold=0.08,
            max_timesteps=2e5,
            eval_freq=2000,
        )
    elif env_name == 'lunar':
        env_specific_params = dict(
            goal_threshold=0.08,
            max_timesteps=2e5,
            eval_freq=2000,
        )
    else:
        raise NotImplementedError(f"Environment {env_name} not recognized")
    
    base_params.update(env_specific_params)
    return base_params


__all__ = [
    # Core classes
    'ModernMujocoEnv',
    'GoalEnv',
    'GymGoalEnvWrapper',
    # Wrappers
    'DiscretizedActionEnv',
    'Discretized',
    # Utils
    'Serializable',
    'MultitaskEnv',
    # Environments
    'SawyerPushGoalEnv',
    'LunarEnv',
    # Factory functions
    'create_env',
    'get_env_params',
    'ENV_NAMES',
    # External environment availability flags
    'METAWORLD_AVAILABLE',
    'GYMNASIUM_ROBOTICS_AVAILABLE',
]
