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
        achieved_goal_space = env.observation_space[achieved_goal_key]
        
        obs_dim = obs_space.shape[0]
        goal_dim = goal_space.shape[0]
        achieved_goal_dim = achieved_goal_space.shape[0]
        
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
        # State now includes: [obs, desired_goal, achieved_goal]
        # This allows goal_distance to extract achieved goals properly
        self.state_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim + goal_dim + achieved_goal_dim,), 
            dtype=np.float32
        )
        self.action_space = env.action_space
        self.achieved_goal_dim = achieved_goal_dim
        
        # Store current goal and achieved goal for step
        self._current_goal = None
        self._current_achieved_goal = None
        self._last_obs_dict = None
    
    def _dict_to_state(self, obs_dict):
        """
        Convert dict observation to state vector.
        
        State format: [obs, desired_goal, achieved_goal]
        This allows goal_distance to properly extract achieved goals for HER.
        """
        obs = obs_dict['observation'].astype(np.float32)
        desired_goal = obs_dict[self.goal_key].astype(np.float32)
        achieved_goal = obs_dict[self.achieved_goal_key].astype(np.float32)
        
        self._current_goal = desired_goal
        self._current_achieved_goal = achieved_goal
        self._last_obs_dict = obs_dict
        
        return np.concatenate([obs, desired_goal, achieved_goal])
    
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
        """Extract DESIRED goal from state (not achieved goal)."""
        obs_dim = self.observation_space.shape[0]
        goal_dim = self.goal_space.shape[0]
        return state[..., obs_dim:obs_dim+goal_dim]
    
    def extract_achieved_goal(self, state):
        """Extract ACHIEVED goal from state."""
        obs_dim = self.observation_space.shape[0]
        goal_dim = self.goal_space.shape[0]
        return state[..., obs_dim+goal_dim:]
    
    def sample_goal(self):
        """
        Sample a random goal state for HER.
        
        This resets the environment and uses the achieved goal as the desired goal.
        For HER relabeling, we sample achieved states as new goals.
        """
        result = self.env.reset()
        
        # Handle both old (dict) and new (dict, info) gym API
        if isinstance(result, tuple):
            obs_dict, info = result
        else:
            obs_dict = result
        
        obs = obs_dict['observation'].astype(np.float32)
        # Use achieved goal as BOTH the desired goal (for sampling) and achieved goal
        # This represents a state where the goal was reached
        achieved = obs_dict[self.achieved_goal_key].astype(np.float32)
        
        # State format: [obs, desired_goal, achieved_goal]
        # For sampled goals, desired = achieved (we reached this state)
        return np.concatenate([obs, achieved, achieved])
    
    def goal_distance(self, state1, state2):
        """
        Compute distance between achieved goal in state1 and desired goal in state2.
        
        Now that state includes [obs, desired_goal, achieved_goal], we can directly
        extract both for proper distance computation.
        
        Args:
            state1: Current state [obs, desired_goal, achieved_goal]
            state2: Goal state [obs, desired_goal, achieved_goal]
        
        Returns:
            Distance between achieved goal in state1 and desired goal in state2
        """
        # Extract achieved goal from state1 (what we actually accomplished)
        achieved_goal = self.extract_achieved_goal(state1)
        
        # Extract desired goal from state2 (what we're trying to reach)
        desired_goal = self.extract_achieved_goal(state2)
        
        return np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    
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

