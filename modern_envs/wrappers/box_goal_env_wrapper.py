"""
General wrapper for environments with Box observation spaces that have embedded goals.

This wrapper extracts goal information from flat Box observations to make them
compatible with goal-conditioned RL algorithms like HDM.

Works for:
- Metaworld environments (goal in last N dimensions)
- Adroit Hand environments (goal based on object/door state)
"""

import gym
import numpy as np
from gym.spaces import Box


class BoxGoalEnvWrapper(gym.Env):
    """
    Wrapper for Box observation space environments to make them HDM-compatible.
    
    For environments where the goal is embedded in the observation as specific indices.
    """
    
    def __init__(self, env, goal_indices, achieved_goal_indices=None, obs_indices=None):
        """
        Args:
            env: The base environment with Box observation space
            goal_indices: Indices in the observation that represent the desired goal
            achieved_goal_indices: Indices representing the currently achieved goal
                                  (if None, uses goal_indices - assumes goal is at end)
            obs_indices: Indices for the observation (if None, uses everything except goal)
        """
        self.env = env
        self.goal_indices = np.array(goal_indices)
        
        total_dim = env.observation_space.shape[0]
        
        # If achieved_goal_indices not specified, assume it's same as goal (typical for Metaworld)
        if achieved_goal_indices is None:
            self.achieved_goal_indices = self.goal_indices
        else:
            self.achieved_goal_indices = np.array(achieved_goal_indices)
        
        # If obs_indices not specified, use everything except the goal indices
        if obs_indices is None:
            all_indices = np.arange(total_dim)
            obs_mask = np.ones(total_dim, dtype=bool)
            obs_mask[self.goal_indices] = False
            self.obs_indices = all_indices[obs_mask]
        else:
            self.obs_indices = np.array(obs_indices)
        
        # Map achieved_goal_indices from full obs to state (obs part)
        # This is needed for goal_distance to extract achieved goals from states
        self.achieved_goal_in_state_indices = []
        for ag_idx in self.achieved_goal_indices:
            # Find where this index appears in obs_indices
            matches = np.where(self.obs_indices == ag_idx)[0]
            if len(matches) > 0:
                self.achieved_goal_in_state_indices.append(matches[0])
        self.achieved_goal_in_state_indices = np.array(self.achieved_goal_in_state_indices)
        
        goal_dim = len(self.goal_indices)
        obs_dim = len(self.obs_indices)
        
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
            shape=(obs_dim + goal_dim,),  # obs + goal
            dtype=np.float32
        )
        self.action_space = env.action_space
        
        # Current goal
        self._current_goal = None
        self._last_full_obs = None
    
    def reset(self, **kwargs):
        """Reset environment and return state (obs + goal)."""
        result = self.env.reset(**kwargs)
        
        # Handle new gym API (returns obs, info)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
        
        self._last_full_obs = obs.astype(np.float32)
        
        # Extract components
        observation = self._last_full_obs[self.obs_indices]
        goal = self._last_full_obs[self.goal_indices]
        
        # State = obs + goal
        state = np.concatenate([observation, goal])
        self._current_goal = goal
        
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
        
        self._last_full_obs = obs.astype(np.float32)
        
        # Extract components
        observation = self._last_full_obs[self.obs_indices]
        goal = self._last_full_obs[self.goal_indices]
        
        # State = obs + goal
        state = np.concatenate([observation, goal])
        self._current_goal = goal
        
        return state, reward, done, info
    
    def observation(self, state):
        """Extract observation from state (everything except goal)."""
        obs_dim = self.observation_space.shape[0]
        return state[..., :obs_dim]
    
    def extract_goal(self, state):
        """Extract goal from state (last goal_dim dimensions)."""
        goal_dim = self.goal_space.shape[0]
        return state[..., -goal_dim:]
    
    def sample_goal(self):
        """
        Sample a random goal state for HER.
        
        Reset the environment to get a new configuration.
        """
        result = self.env.reset()
        
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
        
        full_obs = obs.astype(np.float32)
        observation = full_obs[self.obs_indices]
        goal = full_obs[self.goal_indices]
        
        # Return state = obs + goal
        return np.concatenate([observation, goal])
    
    def goal_distance(self, state1, state2):
        """
        Compute distance between achieved goal in state1 and desired goal in state2.
        
        This measures: How close did we get (achieved) to the target (desired)?
        - state1: Current state (extract achieved goal from observation part)
        - state2: Goal state (extract desired goal)
        """
        # Extract observation from state1
        obs1 = self.observation(state1)
        
        # Extract achieved goal from obs1 using mapped indices
        if len(self.achieved_goal_in_state_indices) > 0:
            # Extract achieved goal from observation part of state
            if state1.ndim == 1:
                achieved_goal = obs1[self.achieved_goal_in_state_indices]
            else:
                achieved_goal = obs1[..., self.achieved_goal_in_state_indices]
        else:
            # If achieved goal indices weren't in obs_indices, fall back to comparing desired goals
            # This is a fallback for incorrectly configured wrappers
            achieved_goal = self.extract_goal(state1)
        
        # Extract desired goal from state2
        desired_goal = self.extract_goal(state2)
        
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





