"""
Wrapper for dict-based goal-conditioned environments (Gymnasium-Robotics, etc.).

Follows the GCSL gymenv_wrapper.py pattern:
- Converts dict observations to state format: [observation, goal, sgoal]
- Both goal and sgoal are achieved_goal by default (duplicated)
- goal_distance compares sgoal (achieved_goal) between states
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
    Wrapper for dict-based goal-conditioned environments.
    
    Follows GCSL pattern:
    - Input: dict obs = {'observation': ..., 'achieved_goal': ..., 'desired_goal': ...}
    - Output: state = [observation, achieved_goal, achieved_goal]
    - goal_distance: compares achieved_goals between states
    """
    
    def __init__(self, env, observation_key='observation', goal_key='achieved_goal', 
                 state_goal_key='achieved_goal', use_internal_rewards=False):
        """
        Args:
            env: Base environment with Dict observation_space
            observation_key: Key for observation in dict (default: 'observation')
            goal_key: Key for goal in dict (default: 'achieved_goal')
            state_goal_key: Key for state goal in dict (default: 'achieved_goal')
            use_internal_rewards: Use env.compute_reward if available
        """
        self.base_env = env
        self.action_space = env.action_space
        
        # Verify dict observation space
        is_dict_space = False
        if GymDict is not None and isinstance(env.observation_space, GymDict):
            is_dict_space = True
        if GymnasiumDict is not None and isinstance(env.observation_space, GymnasiumDict):
            is_dict_space = True
        
        if not is_dict_space:
            raise ValueError(f"Environment must have Dict observation_space, got {type(env.observation_space)}")
        
        # Extract spaces from dict
        all_space = env.observation_space
        
        self.obs_key = observation_key
        self.observation_space = all_space.spaces[observation_key]
        self.goal_key = goal_key
        self.goal_space = all_space.spaces[goal_key]
        self.sgoal_key = state_goal_key
        self.sgoal_space = all_space.spaces[state_goal_key]
        
        # Concatenate observation, goal, and sgoal to get state_space
        obs_low = self.observation_space.low.flatten()
        goal_low = self.goal_space.low.flatten()
        sgoal_low = self.sgoal_space.low.flatten()
        state_low = np.r_[obs_low, goal_low, sgoal_low]
        
        obs_high = self.observation_space.high.flatten()
        goal_high = self.goal_space.high.flatten()
        sgoal_high = self.sgoal_space.high.flatten()
        state_high = np.r_[obs_high, goal_high, sgoal_high]
        
        self.state_space = Box(low=state_low, high=state_high, dtype=np.float32)
        
        self.obs_dims = obs_low.shape[0]
        self.goal_dims = goal_low.shape[0]
        self.sgoal_dims = sgoal_low.shape[0]
        
        self.use_internal_rewards = use_internal_rewards
    
    def _base_obs_to_state(self, base_obs):
        """
        Convert dict observation to state vector.
        
        State format: [observation, goal, sgoal]
        where goal = sgoal = achieved_goal (both from base_obs)
        """
        obs = base_obs[self.obs_key].flatten()
        goal = base_obs[self.goal_key].flatten()
        sgoal = base_obs[self.sgoal_key].flatten()
        return np.r_[obs, goal, sgoal]
    
    def reset(self, **kwargs):
        """Reset environment and return state."""
        result = self.base_env.reset(**kwargs)
        
        # Handle both old (dict) and new (dict, info) gym API
        if isinstance(result, tuple):
            base_obs, info = result
        else:
            base_obs = result
        
        return self._base_obs_to_state(base_obs)
    
    def render(self):
        """Render the environment."""
        return self.base_env.render()
    
    def step(self, action):
        """
        Step environment and return state, reward, done, info.
        
        Returns 4 values (old gym API) for HDM compatibility.
        """
        result = self.base_env.step(action)
        
        # Handle both old (4) and new (5) gym API
        if len(result) == 5:
            base_obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            base_obs, reward, done, info = result
        
        state = self._base_obs_to_state(base_obs)
        
        # Store original obs in info
        info['observation'] = base_obs
        
        return state, reward, done, info
    
    def observation(self, state):
        """
        Extract observation from state.
        
        Args:
            state: [observation, goal, sgoal]
        Returns:
            observation
        """
        obs = state[..., :self.obs_dims]
        return obs.reshape(obs.shape[:len(obs.shape)-1] + self.observation_space.shape)
    
    def extract_goal(self, state):
        """
        Extract goal from state.
        
        Args:
            state: [observation, goal, sgoal]
        Returns:
            goal (achieved_goal)
        """
        goal = state[..., self.obs_dims:self.obs_dims+self.goal_dims]
        return goal.reshape(goal.shape[:len(goal.shape)-1] + self.goal_space.shape)
    
    def _extract_sgoal(self, state):
        """
        Extract sgoal (state goal) from state.
        
        Args:
            state: [observation, goal, sgoal]
        Returns:
            sgoal (achieved_goal)
        """
        sgoal = state[..., self.obs_dims+self.goal_dims:]
        return sgoal.reshape(sgoal.shape[:len(sgoal.shape)-1] + self.sgoal_space.shape)
    
    def sample_goal(self):
        """
        Sample a random goal state for HER episode collection.
        Following GCSL pattern: reset to get new desired_goal
        
        Returns:
            goal_state: State where object is at desired position
        """
        # Get keys for desired_goal
        desired_key = self.goal_key.replace('achieved', 'desired')
        desired_state_key = self.sgoal_key.replace('achieved', 'desired')
        
        result = self.base_env.reset()
        
        if isinstance(result, tuple):
            base_obs, info = result
        else:
            base_obs = result
        
        # Placeholder obs (not used for goal distance)
        obs = (10 + self.observation_space.sample()).flatten()
        
        # Use desired_goal as both goal and sgoal
        goal = base_obs[desired_key].flatten()
        sgoal = base_obs[desired_state_key].flatten()
        
        return np.r_[obs, goal, sgoal]
    
    def goal_distance(self, state, goal_state):
        """
        Compute distance between between achieved goal and (relabeled) desired goal.
        
        Uses sgoal (state_goal) for distance computation.
        Optionally uses env.compute_reward if available.
        
        Args:
            state: Current state
            goal_state: Goal state
        Returns:
            distances: L2 distance or custom reward-based distance
        """
        state_sgoal = self._extract_sgoal(state)
        goal_sgoal = self._extract_sgoal(goal_state)
        
        if self.use_internal_rewards and hasattr(self.base_env, 'compute_reward'):
            # Use environment's reward function (e.g., Gymnasium-Robotics)
            distances = np.abs(np.array([
                self.base_env.compute_reward(achieved, desired, dict())
                for achieved, desired in zip(state_sgoal, goal_sgoal)
            ]))
        else:
            # Use L2 distance
            distances = np.linalg.norm(state_sgoal - goal_sgoal, axis=-1)
        
        return distances
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute reward (optional, for compatibility).
        """
        if hasattr(self.base_env, 'compute_reward'):
            return self.base_env.compute_reward(achieved_goal, desired_goal, info)
        else:
            distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            return -(distance > 0.05).astype(np.float32)
    
    def seed(self, seed=None):
        """Set random seed."""
        if hasattr(self.base_env, 'seed'):
            return self.base_env.seed(seed)
        elif hasattr(self.base_env, 'reset') and seed is not None:
            self.base_env.reset(seed=seed)
            return [seed]
        return [seed]
    
    def close(self):
        """Close the environment."""
        return self.base_env.close()
