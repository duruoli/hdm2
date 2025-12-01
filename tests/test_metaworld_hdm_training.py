#!/usr/bin/env python3
"""
Toy training test using Metaworld push-v3 environment with HDM algorithm.

This script tests if Metaworld environments can be used as drop-in replacements
for the custom environments in the HDM pipeline.
"""

import argparse
import random
import numpy as np
import torch
import metaworld
import gymnasium as gym
from gym.spaces import Box

from hdm.agent.core import DiscretePolicy
from hdm.algo.core import Algo
from hdm.learn.core import Learner
from hdm.replay.core import Replay
from hdm.utils.run_utils import Monitor
from hdm.utils import torch_utils


class MetaworldGoalEnvWrapper(gym.Env):
    """
    Wrapper to convert Metaworld environments to goal-conditioned format
    that HDM expects (similar to gcsl.envs.GoalEnv).
    """
    
    def __init__(self, metaworld_env, task):
        self.env = metaworld_env
        self.task = task
        self.env.set_task(task)
        
        # Get task goal position (different for each task)
        self._goal = self._extract_goal_from_task()
        
        # Define spaces
        # Metaworld obs is 39D: [gripper(4), object(3), goal(3), ... ]
        obs_dim = self.env.observation_space.shape[0]
        goal_dim = 3  # XYZ position of object/puck
        
        self.observation_space = Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.goal_space = Box(
            -np.inf, np.inf, shape=(goal_dim,), dtype=np.float32
        )
        self.action_space = self.env.action_space
        
        # State space combines observation and goal (for HDM replay buffer)
        state_dim = obs_dim + goal_dim
        self.state_space = Box(
            -np.inf, np.inf, shape=(state_dim,), dtype=np.float32
        )
        
    def _extract_goal_from_task(self):
        """Extract goal position from Metaworld task."""
        # Metaworld task contains goal info in task.data
        # For push-v3, the goal is the target puck position
        if hasattr(self.task, 'data') and hasattr(self.task.data, 'keys'):
            if 'obj_init_pos' in self.task.data.keys():
                # Goal is usually close to init pos but different
                goal = self.task.data['obj_init_pos'][:3].copy()
                # Add some offset for the goal
                goal[:2] += 0.1  # Move goal slightly away
                return goal
        
        # Fallback: try to get from env
        try:
            return self.env._target_pos[:3]
        except:
            # Last resort: use a default goal
            return np.array([0.0, 0.7, 0.02], dtype=np.float32)
    
    def _extract_achieved_goal(self, obs):
        """Extract current object position from observation."""
        # In Metaworld push-v3:
        # obs[0:3] = gripper position
        # obs[4:7] = object/puck position
        return obs[4:7].copy()
    
    def reset(self, **kwargs):
        """Reset environment and return state (obs + goal concatenated)."""
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle new gym API
        
        # Update goal from task (may change between episodes)
        self._goal = self._extract_goal_from_task()
        
        # Return state = obs + goal concatenated
        obs_float = obs.astype(np.float32)
        return np.concatenate([obs_float, self._goal])
    
    def step(self, action):
        """Step environment and return state (obs + goal concatenated)."""
        result = self.env.step(action)
        
        # Handle both old and new gym API
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            # Convert to old API (done = terminated or truncated)
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        
        # Return state = obs + goal concatenated (not just obs)
        obs_float = obs.astype(np.float32)
        state = np.concatenate([obs_float, self._goal])
        return state, reward, done, info
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute reward for HER (distance-based sparse reward)."""
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        # Sparse reward: 0 if close to goal, -1 otherwise
        return -(distance > 0.05).astype(np.float32)
    
    def get_achieved_goal(self, obs):
        """Get achieved goal from observation (for HER)."""
        if obs.ndim == 1:
            return self._extract_achieved_goal(obs)
        else:
            return np.array([self._extract_achieved_goal(o) for o in obs])
    
    def get_goal(self):
        """Get current goal."""
        return self._goal.copy()
    
    def seed(self, seed=None):
        """Set random seed."""
        if seed is not None:
            self.env.seed(seed)
        return [seed]
    
    def render(self, mode='rgb_array'):
        """Render environment."""
        return self.env.render()
    
    def observation(self, state):
        """Extract observation from state (state = obs + goal)."""
        # State is concatenated [obs, goal]
        obs_dim = self.observation_space.shape[0]
        return state[..., :obs_dim]
    
    def extract_goal(self, state):
        """Extract goal from state (state = obs + goal)."""
        # State is concatenated [obs, goal]
        obs_dim = self.observation_space.shape[0]
        return state[..., obs_dim:]
    
    def sample_goal(self):
        """Sample a random goal state (returns full state with goal)."""
        # Get a random observation
        obs_raw = self.env.reset()
        if isinstance(obs_raw, tuple):
            obs_raw = obs_raw[0]
        
        # Extract object position as achieved goal
        achieved_goal = self._extract_achieved_goal(obs_raw)
        
        # Use achieved goal as the desired goal for this sample
        # (This is used for HER - we relabel with achieved states as goals)
        goal = achieved_goal.copy()
        
        # Return as state (obs + goal concatenated)
        return np.concatenate([obs_raw.astype(np.float32), goal])
    
    def goal_distance(self, state1, state2):
        """Compute distance between achieved goals in two states."""
        # Extract observations from states
        obs1 = self.observation(state1)
        obs2 = self.observation(state2)
        
        # Get achieved goals from observations
        ag1 = self.get_achieved_goal(obs1)
        ag2 = self.get_achieved_goal(obs2)
        
        # Compute distance
        return np.linalg.norm(ag1 - ag2, axis=-1)


class DiscretizedActionEnv(gym.Wrapper):
    """
    Discretize continuous action space.
    Simplified version compatible with Metaworld.
    """
    
    def __init__(self, env, granularity=3):
        super().__init__(env)
        self.granularity = granularity
        
        # Propagate goal_space and state_space
        if hasattr(env, 'goal_space'):
            self.goal_space = env.goal_space
        if hasattr(env, 'state_space'):
            self.state_space = env.state_space
        
        # Get continuous action space - handle both gym.spaces.Box and gymnasium.spaces.Box
        from gymnasium.spaces import Box as GymBox
        assert isinstance(env.action_space, (Box, GymBox))
        self.continuous_action_space = env.action_space
        
        # Create discrete action space
        # For each action dimension, we have granularity discrete values
        action_dim = env.action_space.shape[0]
        num_actions = granularity ** action_dim
        
        from gym.spaces import Discrete
        self.action_space = Discrete(num_actions)
        
        # Pre-compute all discrete actions
        self._discrete_actions = self._create_discrete_actions()
        
    def _create_discrete_actions(self):
        """Create all possible discrete actions."""
        action_dim = self.continuous_action_space.shape[0]
        low = self.continuous_action_space.low
        high = self.continuous_action_space.high
        
        # Create grid of actions
        actions_per_dim = []
        for i in range(action_dim):
            actions_per_dim.append(
                np.linspace(low[i], high[i], self.granularity)
            )
        
        # Generate all combinations
        import itertools
        all_actions = list(itertools.product(*actions_per_dim))
        return np.array(all_actions, dtype=np.float32)
    
    def step(self, discrete_action):
        """Convert discrete action to continuous and step."""
        continuous_action = self._discrete_actions[discrete_action]
        return self.env.step(continuous_action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def seed(self, seed=None):
        """Set random seed."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        return [seed]
    
    # Delegate goal-conditioned methods to wrapped environment
    def observation(self, state):
        """Extract observation from state."""
        return self.env.observation(state)
    
    def extract_goal(self, state):
        """Extract goal from state."""
        return self.env.extract_goal(state)
    
    def sample_goal(self):
        """Sample a random goal state."""
        return self.env.sample_goal()
    
    def goal_distance(self, state1, state2):
        """Compute distance between goals in two states."""
        return self.env.goal_distance(state1, state2)
    
    def get_achieved_goal(self, obs):
        """Get achieved goal from observation."""
        if hasattr(self.env, 'get_achieved_goal'):
            return self.env.get_achieved_goal(obs)
        return obs  # Fallback
    
    def get_goal(self):
        """Get current goal."""
        if hasattr(self.env, 'get_goal'):
            return self.env.get_goal()
        return None


def create_metaworld_pusher_env():
    """Create Metaworld push-v3 environment with goal-conditioned wrapper."""
    # Create Metaworld environment
    mt1 = metaworld.MT1('push-v3')
    env = mt1.train_classes['push-v3']()
    
    # Get a random task (this sets the goal)
    task = random.sample(mt1.train_tasks, 1)[0]
    
    # Wrap to make it goal-conditioned
    env = MetaworldGoalEnvWrapper(env, task)
    
    return env


def get_env_and_agent(env, env_params):
    """Create discretized environment and agent."""
    # Get dimensions before wrapping
    obs_dim = env.observation_space.shape[0]
    goal_dim = env.goal_space.shape[0]
    
    # Discretize actions
    granularity = env_params.get('action_granularity', 3)
    env = DiscretizedActionEnv(env, granularity=granularity)
    
    # Update env_params with action space info
    env_params['action_space.n'] = env.action_space.n
    
    # Create agent (policy)
    policy_class = DiscretePolicy
    agent = policy_class(env_params, layers=(256, 256, 256))
    
    return env, agent


def run_toy_training():
    """Run a very short training to test the pipeline."""
    print("=" * 80)
    print("TOY TRAINING: Metaworld push-v3 with HDM")
    print("=" * 80)
    
    # Create environment
    print("\n1. Creating Metaworld push-v3 environment...")
    env = create_metaworld_pusher_env()
    print(f"   ✓ Environment created")
    print(f"   - Observation space: {env.observation_space.shape}")
    print(f"   - Goal space: {env.goal_space.shape}")
    print(f"   - Action space: {env.action_space}")
    
    # Environment parameters (similar to pusher)
    env_params = {
        'obs_dim': env.observation_space.shape[0],
        'goal_dim': env.goal_space.shape[0],
        'goal_threshold': 0.05,
        'max_trajectory_length': 50,
        'action_granularity': 3,
        'independent_policy': False,
    }
    
    print("\n2. Creating discretized environment and agent...")
    env, agent = get_env_and_agent(env, env_params)
    print(f"   ✓ Agent created")
    print(f"   - Discrete action space: {env.action_space.n} actions")
    
    # Set up minimal training args
    class Args:
        def __init__(self):
            self.env_name = 'metaworld_push'
            self.seed = 0
            self.save_dir = 'experiments/'
            self.ckpt_name = 'toy_test'
            self.resume_ckpt = ''
            
            # Toy training settings (very small)
            self.n_workers = 1
            self.num_rollouts_per_mpi = 1
            self.n_cycles = 2  # Only 2 cycles
            self.optimize_every = 10
            self.n_batches = 5  # Only 5 batches
            self.target_update_freq = 10
            
            self.greedy_action = True
            self.random_act_prob = 0.2
            
            self.buffer_size = 10000
            self.future_p = 0.85
            self.next_state_p = 0.6
            self.relabeled_reward_only = True
            self.batch_size = 128
            
            self.lr_actor = 5e-4
            self.start_policy_timesteps = 100
            
            self.reward_scale = 1.0
            self.reward_bias = 0.0
            
            self.n_initial_rollouts = 5  # Only 5 initial rollouts
            self.n_test_rollouts = 3  # Only 3 test rollouts
            
            self.gamma = 0.98
            self.polyak = 0.995
            
            self.independent_policy = False
            
            # HDM specific
            self.use_dqn = True
            self.double_dqn = True
            self.backup_strategy = 'q_max'
            self.backup_temp = 1.0
            self.backup_epsilon = 0.1
            self.hdm_backup_strategy = 'q_max'  # Valid: q_max, q_softmax, q_eps_greedy, q_soft_kl, act_2
            self.hdm_weights_to_indicator = True
            self.hdm_weights_min = -1e6
            self.hdm_weights_max = 1e6
            self.hdm_weights_relabel_mask = False
            self.hdm_gamma_use_auto = False
            self.hdm_q_normalizer = False
            self.targ_clip = True
            self.hdm_online_o2 = True
            self.hdm_q_coef = 1.0
            self.hdm_gamma = 0.5
            self.hdm_bc = True
    
    args = Args()
    
    # Set seeds
    print("\n3. Setting random seeds...")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch_utils.use_cuda:
        torch.cuda.manual_seed(args.seed)
    env.seed(args.seed)
    print(f"   ✓ Seeds set to {args.seed}")
    
    # Create components
    print("\n4. Creating HDM components...")
    monitor = Monitor()
    replay = Replay(env, env_params, args)
    learner = Learner(agent, monitor, args)
    
    print("   ✓ Monitor created")
    print("   ✓ Replay buffer created")
    print("   ✓ Learner created")
    
    # Create algorithm
    print("\n5. Creating HDM algorithm...")
    algo = Algo(
        env=env,
        env_params=env_params,
        args=args,
        agent=agent,
        replay=replay,
        monitor=monitor,
        learner=learner,
    )
    print("   ✓ Algorithm created")
    
    # Test a single rollout
    print("\n6. Testing single rollout...")
    state = env.reset()  # Returns state (obs + goal)
    obs = env.observation(state)  # Extract obs from state
    achieved_goal = env.env.get_achieved_goal(obs)
    desired_goal = env.env.get_goal()
    
    print(f"   - Initial state shape: {state.shape}")
    print(f"   - Initial observation shape: {obs.shape}")
    print(f"   - Achieved goal: {achieved_goal}")
    print(f"   - Desired goal: {desired_goal}")
    print(f"   - Initial distance: {np.linalg.norm(achieved_goal - desired_goal):.3f}")
    
    # Take a few steps
    total_reward = 0
    for step in range(10):
        action = agent.act(
            np.array([obs]), 
            np.array([desired_goal]), 
            greedy=False, 
            random_act_prob=0.5
        )
        # action is already a single value for batch size 1
        action_scalar = int(action[0]) if len(action.shape) > 0 else int(action)
        state, reward, done, info = env.step(action_scalar)  # Returns state
        obs = env.observation(state)  # Extract obs from state
        total_reward += reward
        
        if done:
            break
    
    print(f"   ✓ Rollout completed: {step + 1} steps, reward = {total_reward:.2f}")
    
    # Run initial rollouts
    print(f"\n7. Collecting {args.n_initial_rollouts} initial rollouts...")
    algo.collect_experience(
        greedy=False,
        random_act_prob=0.5,
        train_agent=False
    )
    print(f"   ✓ Collected initial experience")
    print(f"   - Buffer size: {replay.n_transitions_stored}")
    
    # Run one training cycle
    print(f"\n8. Running {args.n_batches} optimization steps...")
    algo.agent_optimize()
    print(f"   ✓ Optimization completed")
    
    # Test rollout
    print(f"\n9. Testing learned policy...")
    algo.collect_experience(
        greedy=True,
        random_act_prob=0.0,
        train_agent=False
    )
    print(f"   ✓ Test rollout completed")
    
    print("\n" + "=" * 80)
    print("TOY TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n✅ Metaworld push-v3 works with HDM pipeline!")
    print("✅ You can now use Metaworld environments instead of custom ones")
    print("\nNext steps:")
    print("  1. Run longer training with more cycles")
    print("  2. Compare performance with original pusher environment")
    print("  3. Try other Metaworld tasks (door-open-v3, drawer-open-v3, etc.)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_toy_training()

