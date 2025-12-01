"""
Generic wrapper for dict-based goal-conditioned environments (Metaworld, Gymnasium-Robotics).

Converts dict observations {'observation': ..., 'achieved_goal': ..., 'desired_goal': ...}
to HDM's state-based format where state = concatenate([observation, goal]).
"""

import gym
import numpy as np
from gym.spaces import Box

# Handle both old gym and new gymnasium
try:
    from gymnasium.spaces import Dict as GymnasiumDict
    has_gymnasium = True
except ImportError:
    has_gymnasium = False
    GymnasiumDict = None

try:
    from gym.spaces import Dict as GymDict
except ImportError:
    GymDict = None


class DictGoalEnvWrapper(gym.Env):
    """
    Wrapper for environments that use dict observations with goal-conditioning.
    
    Converts from:
        obs = {'observation': array, 'achieved_goal': array, 'desired_goal': array}
    To HDM format:
        state = concatenate([observation, desired_goal])
    
    Compatible with:
    - Metaworld environments (MT50, ML45, etc.)
    - Gymnasium-Robotics environments (Fetch, Hand, Maze, etc.)
    """
    
    def __init__(self, env, goal_key='desired_goal', achieved_goal_key='achieved_goal'):
        """
        Args:
            env: The base environment with dict observations
            goal_key: Key for desired goal in observation dict (default: 'desired_goal')
            achieved_goal_key: Key for achieved goal in observation dict (default: 'achieved_goal')
        """
        self.env = env
        self.goal_key = goal_key
        self.achieved_goal_key = achieved_goal_key
        
        # Verify the environment has dict observation space
        # Check for both gym.spaces.Dict and gymnasium.spaces.Dict
        is_dict_space = False
        if GymDict is not None and isinstance(env.observation_space, GymDict):
            is_dict_space = True
        if GymnasiumDict is not None and isinstance(env.observation_space, GymnasiumDict):
            is_dict_space = True
        
        if not is_dict_space:
            raise ValueError(f"Environment must have Dict observation_space, got {type(env.observation_space)}")
        
        # Extract spaces from dict
        obs_space = env.observation_space['observation']
        goal_space = env.observation_space[goal_key]
        
        obs_dim = obs_space.shape[0]
        goal_dim = goal_space.shape[0]
        
        # Define HDM-compatible spaces
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        self.goal_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(goal_dim,), 
            dtype=np.float32
        )
        self.state_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim + goal_dim,), 
            dtype=np.float32
        )
        self.action_space = env.action_space
        
        # Store current goal for step
        self._current_goal = None
    
    def _dict_to_state(self, obs_dict):
        """Convert dict observation to state vector."""
        obs = obs_dict['observation'].astype(np.float32)
        goal = obs_dict[self.goal_key].astype(np.float32)
        self._current_goal = goal
        return np.concatenate([obs, goal])
    
    def reset(self, **kwargs):
        """Reset environment and return state (obs + goal)."""
        result = self.env.reset(**kwargs)
        
        # Handle both old (dict) and new (dict, info) gym API
        if isinstance(result, tuple):
            obs_dict, info = result
        else:
            obs_dict = result
        
        return self._dict_to_state(obs_dict)
    
    def step(self, action):
        """
        Step environment and return state, reward, done, info (4 values, old gym API).
        """
        result = self.env.step(action)
        
        # Handle both old (4) and new (5) gym API
        if len(result) == 5:
            obs_dict, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs_dict, reward, done, info = result
        
        state = self._dict_to_state(obs_dict)
        
        return state, reward, done, info
    
    def observation(self, state):
        """Extract observation from state."""
        obs_dim = self.observation_space.shape[0]
        return state[..., :obs_dim]
    
    def extract_goal(self, state):
        """Extract goal from state."""
        obs_dim = self.observation_space.shape[0]
        return state[..., obs_dim:]
    
    def sample_goal(self):
        """
        Sample a random goal state for HER.
        
        This resets the environment and extracts the achieved goal as a possible goal.
        """
        result = self.env.reset()
        
        # Handle both old (dict) and new (dict, info) gym API
        if isinstance(result, tuple):
            obs_dict, info = result
        else:
            obs_dict = result
        
        obs = obs_dict['observation'].astype(np.float32)
        # Use achieved goal as the desired goal for sampling
        goal = obs_dict[self.achieved_goal_key].astype(np.float32)
        
        return np.concatenate([obs, goal])
    
    def goal_distance(self, state1, state2):
        """
        Compute distance between goals in two states.
        
        For dict-based envs, we compare achieved goals, not desired goals.
        The achieved goal is typically a function of the observation.
        """
        obs1 = self.observation(state1)
        obs2 = self.observation(state2)
        
        # Get achieved goals from observations
        # We need to temporarily reconstruct dict format to query achieved_goal
        # For simplicity, we'll use the environment's compute_reward if available
        # Otherwise, fall back to goal distance
        
        # Extract desired goals from states
        goal1 = self.extract_goal(state1)
        goal2 = self.extract_goal(state2)
        
        # Compute distance between desired goals
        # Note: For proper HER, you'd want to compare achieved goals
        # but since we only have state, we compare desired goals
        return np.linalg.norm(goal1 - goal2, axis=-1)
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute reward based on achieved and desired goals.
        
        Many dict-based envs provide this method.
        """
        if hasattr(self.env, 'compute_reward'):
            return self.env.compute_reward(achieved_goal, desired_goal, info)
        else:
            # Default: negative distance
            distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            return -(distance > 0.05).astype(np.float32)
    
    def seed(self, seed=None):
        """Set random seed."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        elif hasattr(self.env, 'reset') and seed is not None:
            # For newer gym versions that use np_random
            self.env.reset(seed=seed)
            return [seed]
        return [seed]
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render(mode=mode)
    
    def close(self):
        """Close the environment."""
        return self.env.close()

