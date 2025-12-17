"""
Metaworld-specific wrapper for HDM compatibility.

Follows the GCSL gymenv_wrapper.py pattern:
- Wraps MetaWorld V3 flat Box observations into HDM state format
- achieved_goal = [gripper_pos, object_pos] (6D)
- state = [observation, achieved_goal, achieved_goal] (duplicated)
- goal_distance only compares object positions
"""

import gym
import numpy as np
from gym.spaces import Box


class MetaworldGoalWrapper(gym.Env):
    """
    Wrapper for Metaworld V3 environments to make them HDM-compatible.
    
    Follows GCSL sawyer_push pattern:
    - MetaWorld obs: flat 39D vector [gripper_xyz, gripper_state, object_xyz, ...]
    - achieved_goal: [gripper_xyz, object_xyz] = 6D
    - state: [observation, achieved_goal, achieved_goal]
    - goal_distance: compares ONLY object positions (not gripper)
    
    The goal is NOT in MetaWorld's observation - it's stored in env._target_pos.
    """
    
    def __init__(self, env, robot_indices=None, object_indices=None, task_list=None):
        """
        Args:
            env: The base Metaworld environment
            robot_indices: Indices for robot/gripper position in observation.
                          Default: [0, 1, 2] (gripper xyz)
            object_indices: Indices for object position in observation.
                           Default: [4, 5, 6] (object xyz)
            task_list: List of tasks to sample from on each reset (for randomization)
        """
        self.env = env
        self.action_space = env.action_space
        self.task_list = task_list  # Store task list for randomization
        
        # Default indices for MetaWorld
        if robot_indices is None:
            self.robot_indices = [0, 1, 2]  # Gripper xyz position
        else:
            self.robot_indices = robot_indices
            
        if object_indices is None:
            self.object_indices = [4, 5, 6]  # Object xyz position
        else:
            self.object_indices = object_indices
        
        # MetaWorld observation is flat Box
        all_space = env.observation_space
        
        # observation_space = full MetaWorld obs (no goal appended yet)
        self.observation_space = Box(
            low=all_space.low,
            high=all_space.high,
            shape=all_space.shape,
            dtype=np.float32
        )
        
        # achieved_goal = [gripper_xyz, object_xyz] = 6D
        achieved_goal_dim = len(self.robot_indices) + len(self.object_indices)
        
        self.goal_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(achieved_goal_dim,),
            dtype=np.float32
        )
        
        # sgoal_space is same as goal_space
        self.sgoal_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(achieved_goal_dim,),
            dtype=np.float32
        )
        
        # state_space = [observation, goal, sgoal]
        # All three concatenated
        obs_low = self.observation_space.low.flatten()
        goal_low = self.goal_space.low.flatten()
        sgoal_low = self.sgoal_space.low.flatten()
        state_low = np.r_[obs_low, goal_low, sgoal_low]
        
        obs_high = self.observation_space.high.flatten()
        goal_high = self.goal_space.high.flatten()
        sgoal_high = self.sgoal_space.high.flatten()
        state_high = np.r_[obs_high, goal_high, sgoal_high]
        
        self.state_space = Box(low=state_low, high=state_high, dtype=np.float32)
        
        # Store dimensions for extraction
        self.obs_dims = obs_low.shape[0]
        self.goal_dims = goal_low.shape[0]
        self.sgoal_dims = sgoal_low.shape[0]
    
    def _base_obs_to_state(self, base_obs):
        """
        Convert MetaWorld observation to HDM state format.
        
        Args:
            base_obs: MetaWorld observation (39D flat array)
            
        Returns:
            state: [observation, goal, sgoal]
                   where goal = [gripper_xyz, object_xyz]
                   and sgoal = [gripper_xyz, object_xyz]
        """
        obs = base_obs.flatten()
        
        # Extract achieved_goal components
        gripper_pos = obs[self.robot_indices]
        object_pos = obs[self.object_indices]
        achieved_goal = np.concatenate([gripper_pos, object_pos])
        
        # Concatenate: [obs, achieved_goal, achieved_goal]
        # Both goal and sgoal are the same (achieved_goal)
        return np.r_[obs, achieved_goal, achieved_goal]
    
    def reset(self, **kwargs):
        """
        Reset environment and return state.
        
        Returns:
            state: [observation, achieved_goal, achieved_goal]
        """
        # Randomize task if task list is provided
        if self.task_list is not None and len(self.task_list) > 0:
            import random
            task = random.choice(self.task_list)
            self.env.set_task(task)
        
        result = self.env.reset(**kwargs)
        
        # Handle new gym API (returns obs, info)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
        
        obs = obs.astype(np.float32)
        return self._base_obs_to_state(obs)
    
    def step(self, action):
        """
        Step environment and return state, reward, done, info.
        
        Returns 4 values (old gym API) for HDM compatibility.
        """
        result = self.env.step(action)
        
        # Handle new gym API (5 values)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        
        obs = obs.astype(np.float32)
        state = self._base_obs_to_state(obs)
        
        # Store original obs in info for debugging
        info['observation'] = obs
        
        return state, reward, done, info
    
    def observation(self, state):
        """
        Extract observation from state.
        
        Args:
            state: HDM state [observation, goal, sgoal]
            
        Returns:
            observation: The original observation part
        """
        obs = state[..., :self.obs_dims]
        return obs.reshape(obs.shape[:len(obs.shape)-1] + self.observation_space.shape)
    
    def extract_goal(self, state):
        """
        Extract goal from state.
        
        Args:
            state: HDM state [observation, goal, sgoal]
            
        Returns:
            goal: achieved_goal = [gripper_xyz, object_xyz]
        """
        goal = state[..., self.obs_dims:self.obs_dims+self.goal_dims]
        return goal.reshape(goal.shape[:len(goal.shape)-1] + self.goal_space.shape)
    
    def _extract_sgoal(self, state):
        """
        Extract sgoal (state goal) from state.
        
        This is the same as extract_goal - both are achieved_goal.
        
        Args:
            state: HDM state [observation, goal, sgoal]
            
        Returns:
            sgoal: achieved_goal = [gripper_xyz, object_xyz]
        """
        sgoal = state[..., self.obs_dims+self.goal_dims:]
        return sgoal.reshape(sgoal.shape[:len(sgoal.shape)-1] + self.sgoal_space.shape)
    
    def sample_goal(self):
        """
        Sample a random goal state for HER.
        
        Following GCSL pattern: reset to get new desired_goal (env._target_pos),
        then create a state where achieved_goal = desired_goal.
        
        Returns:
            goal_state: A full state where object is at the target position
        """
        result = self.env.reset()
        
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
        
        obs = obs.astype(np.float32)
        
        # Get the desired goal position from environment
        object_target = self.env._target_pos.copy()  # 3D object xyz
        gripper_target = object_target.copy()  # Gripper should be near/at object target (this is psuedo desired gripper position)
        goal_6d = np.concatenate([gripper_target, object_target])  # 6D [gripper_xyz, object_xyz]
        goal_state = np.r_[obs, goal_6d, goal_6d]
        
        
        return goal_state.astype(np.float32)
    
    def goal_distance(self, state, goal_state):
        """
        Compute distance between achieved goal and desired goal.
        
        Following GCSL pattern: only compare OBJECT positions, not gripper.
        
        Args:
            state: Current state
            goal_state: Goal state
            
        Returns:
            distances: L2 distance between object positions
        """
        # Extract achieved goals (both are [gripper_xyz, object_xyz])
        achieved_goal = self._extract_sgoal(state)
        desired_goal = self._extract_sgoal(goal_state)
        
        # Compare ONLY object positions (last 3 elements)
        # achieved_goal = [gripper_xyz, object_xyz] = 6D
        # We want object_xyz which is indices [3:6]
        diff = achieved_goal - desired_goal
        
        # Use only object position for distance (indices 3:6 = object_xyz)
        distances = np.linalg.norm(diff[..., 0:6], axis=-1)
        return distances
    
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
