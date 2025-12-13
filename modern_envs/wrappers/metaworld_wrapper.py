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
    - obs[0:3] = hand/gripper position
    - obs[3] = gripper state
    - obs[4:7] = object position (ACHIEVED GOAL - what robot actually moved)
    - ... other state info ...
    - obs[-3:] = desired goal position (TARGET)
    
    For success measurement, we compare ACHIEVED goal (object position) 
    with DESIRED goal (target position).
    """
    
    def __init__(self, env, goal_dim=3, achieved_goal_indices=None):
        """
        Args:
            env: The base Metaworld environment
            goal_dim: Dimension of the goal (default: 3 for xyz position)
            achieved_goal_indices: Indices for achieved goal in observation.
                                   Default: [4, 5, 6] (object position)
        """
        self.env = env
        self.goal_dim = goal_dim
        
        # Metaworld standard observation structure:
        # obs[4:7] = object/puck position (achieved goal)
        # obs[-3:] = desired goal position
        if achieved_goal_indices is None:
            self.achieved_goal_indices = [4, 5, 6]  # Object position
        else:
            self.achieved_goal_indices = achieved_goal_indices
        
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
        """Extract DESIRED goal from state (last goal_dim dimensions)."""
        return state[..., -self.goal_dim:]
    
    def extract_achieved_goal(self, state):
        """
        Extract ACHIEVED goal from state (object position).
        
        This is typically the object/puck position that the robot manipulates,
        NOT the desired target position.
        """
        if state.ndim == 1:
            return state[self.achieved_goal_indices].copy()
        else:
            # Batched states
            return state[..., self.achieved_goal_indices].copy()
    
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
        Compute distance between ACHIEVED goal in state1 and DESIRED goal in state2.
        
        This measures how close the robot got (achieved) to the target (desired).
        - state1: Current state (we extract achieved goal = object position)
        - state2: Goal state (we extract desired goal = target position)
        
        Returns:
            Distance between achieved goal and desired goal
        """
        # Achieved goal = where the object actually is (from state1)
        achieved_goal = self.extract_achieved_goal(state1)
        # Desired goal = where we want the object to be (from state2)
        desired_goal = self.extract_achieved_goal(state2)
        
        return np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    
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


