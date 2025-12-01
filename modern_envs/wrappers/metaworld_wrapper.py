"""
Metaworld-specific wrapper for HDM compatibility.

Metaworld V3 environments use flat Box observations (not Dict),
so they need a different wrapper than Gymnasium-Robotics.
"""

import gym
import numpy as np
from gym.spaces import Box


class MetaworldGoalWrapper(gym.Env):
    """
    Wrapper for Metaworld V3 environments to make them HDM-compatible.
    
    Metaworld observations are flat vectors where:
    - The goal is embedded in the observation (typically last 3 dimensions)
    - We need to extract and expose it properly for HDM
    """
    
    def __init__(self, env, goal_dim=3):
        """
        Args:
            env: The base Metaworld environment
            goal_dim: Dimension of the goal (default: 3 for xyz position)
        """
        self.env = env
        self.goal_dim = goal_dim
        
        # Metaworld observation includes the goal, but we need to separate them
        # Total obs = robot_state + object_state + goal
        total_dim = env.observation_space.shape[0]
        obs_dim = total_dim - goal_dim
        
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
            shape=(total_dim,),  # obs + goal
            dtype=np.float32
        )
        self.action_space = env.action_space
        
        # Current goal
        self._current_goal = None
    
    def reset(self, **kwargs):
        """Reset environment and return state (obs + goal)."""
        result = self.env.reset(**kwargs)
        
        # Handle new gym API (returns obs, info)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
        
        # Metaworld already concatenates obs + goal
        # Extract goal and store it
        state = obs.astype(np.float32)
        self._current_goal = state[-self.goal_dim:]
        
        return state
    
    def step(self, action):
        """Step environment and return state, reward, done, info (4 values, old gym API)."""
        result = self.env.step(action)
        
        # Handle new gym API (5 values)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        
        state = obs.astype(np.float32)
        self._current_goal = state[-self.goal_dim:]
        
        return state, reward, done, info
    
    def observation(self, state):
        """Extract observation from state (everything except goal)."""
        obs_dim = self.observation_space.shape[0]
        return state[..., :obs_dim]
    
    def extract_goal(self, state):
        """Extract goal from state (last goal_dim dimensions)."""
        return state[..., -self.goal_dim:]
    
    def sample_goal(self):
        """
        Sample a random goal state for HER.
        
        For Metaworld, we reset the environment to get a new configuration.
        The goal position is embedded in the observation.
        """
        result = self.env.reset()
        
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
        
        state = obs.astype(np.float32)
        return state
    
    def goal_distance(self, state1, state2):
        """
        Compute distance between goals in two states.
        
        For Metaworld, we compare the goal positions (last 3 dimensions).
        """
        goal1 = self.extract_goal(state1)
        goal2 = self.extract_goal(state2)
        return np.linalg.norm(goal1 - goal2, axis=-1)
    
    def seed(self, seed=None):
        """Set random seed."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        elif hasattr(self.env, 'reset') and seed is not None:
            # For newer gym versions
            self.env.reset(seed=seed)
            return [seed]
        return [seed]
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        return self.env.close()


