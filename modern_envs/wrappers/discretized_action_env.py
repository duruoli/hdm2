"""
DiscretizedActionEnv - converts continuous actions to discrete grid.

HDM algorithm requires discrete actions, so this wrapper discretizes
the continuous action space into a grid.

Copied from GCSL to avoid dependencies.
"""

import numpy as np
from gym.spaces import Discrete


class Discretized(Discrete):
    """
    Discrete action space with metadata about discretization.
    
    Extends gym.spaces.Discrete to expose:
    - n_dims: number of action dimensions
    - granularity: number of discretization levels per dimension
    """
    def __init__(self, n, n_dims, granularity):
        self.n_dims = n_dims
        self.granularity = granularity
        assert n == granularity ** n_dims
        super(Discretized, self).__init__(n)


class DiscretizedActionEnv:
    """
    Wrapper that discretizes continuous action space into a grid.
    
    For example, a 2D continuous action space with granularity=3:
    - Creates 3^2 = 9 discrete actions
    - Each action maps to a point on the grid: 
      [(low, low), (low, mid), (low, high), 
       (mid, low), (mid, mid), (mid, high),
       (high, low), (high, mid), (high, high)]
    
    This is needed because HDM uses discrete Q-learning.
    """
    
    def __init__(self, wrapped_env, possible_actions=None, granularity=3):
        """
        Initialize discretized action wrapper.
        
        Args:
            wrapped_env: Environment with continuous action space (or already discrete)
            possible_actions: Optional pre-defined action set (if None, creates grid)
            granularity: Number of levels per dimension (if possible_actions is None)
        """
        self.wrapped_env = wrapped_env
        
        # Check if already discrete
        if isinstance(wrapped_env.action_space, Discrete):
            # Already discrete, pass through
            self.action_space = wrapped_env.action_space
            self.base_actions = None  # Not needed for discrete actions
            self.is_discrete = True
        elif possible_actions is not None:
            self.base_actions = possible_actions
            n_dims = 1
            granularity = len(self.base_actions)
            self.action_space = Discretized(len(self.base_actions), n_dims, granularity)
            self.is_discrete = False
        else:
            # Create grid of actions
            actions_meshed = np.meshgrid(*[
                np.linspace(lo, hi, granularity) 
                for lo, hi in zip(
                    self.wrapped_env.action_space.low, 
                    self.wrapped_env.action_space.high
                )
            ])
            self.base_actions = np.array([a.flat[:] for a in actions_meshed]).T
            n_dims = self.wrapped_env.action_space.shape[0]
            self.action_space = Discretized(len(self.base_actions), n_dims, granularity)
            self.is_discrete = False
        
        # Pass through other attributes
        self.observation_space = self.wrapped_env.observation_space
        if hasattr(self.wrapped_env, 'goal_space'):
            self.goal_space = self.wrapped_env.goal_space
        if hasattr(self.wrapped_env, 'state_space'):
            self.state_space = self.wrapped_env.state_space

    def step(self, action):
        """
        Take a step with discretized action.
        
        Args:
            action: Discrete action index (int)
            
        Returns:
            Same as wrapped environment
        """
        if self.is_discrete:
            # Already discrete, pass through
            return self.wrapped_env.step(action)
        else:
            return self.wrapped_env.step(self.base_actions[action])

    def reset(self):
        """Reset environment."""
        return self.wrapped_env.reset()
    
    def render(self, *args, **kwargs):
        """Render environment."""
        return self.wrapped_env.render(*args, **kwargs)
    
    def seed(self, seed=None):
        """Set random seed."""
        return self.wrapped_env.seed(seed)
    
    # Pass through goal-conditioned methods
    def observation(self, state):
        return self.wrapped_env.observation(state)
    
    def extract_goal(self, state):
        return self.wrapped_env.extract_goal(state)
    
    def goal_distance(self, state, goal_state):
        return self.wrapped_env.goal_distance(state, goal_state)
    
    def sample_goal(self):
        return self.wrapped_env.sample_goal()
    
    def get_diagnostics(self, trajectories, desired_goal_states):
        return self.wrapped_env.get_diagnostics(trajectories, desired_goal_states)
    
    def __getattr__(self, name):
        """Pass through any other attributes to wrapped env."""
        return getattr(self.wrapped_env, name)

