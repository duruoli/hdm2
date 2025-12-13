import argparse
import os
import random
import time

import gym
import numpy as np
import torch
from gym.spaces import Discrete

# Try to use modern environments first, fallback to gcsl
try:
    # #region agent log
    import json,time;open('/Users/duruoli/A/A李杜若/1-科研/PhD/0/1-code/hdm2/.cursor/debug.log','a').write(json.dumps({'sessionId':'debug-session','runId':'pre-fix','hypothesisId':'H1,H2,H3','location':'__main__.py:12','message':'Attempting modern_envs import','data':{},'timestamp':int(time.time()*1000)})+'\n')
    # #endregion
    import modern_envs as envs
    from modern_envs.wrappers import DiscretizedActionEnv
    # #region agent log
    import json,time;open('/Users/duruoli/A/A李杜若/1-科研/PhD/0/1-code/hdm2/.cursor/debug.log','a').write(json.dumps({'sessionId':'debug-session','runId':'pre-fix','hypothesisId':'H1','location':'__main__.py:16','message':'modern_envs imported successfully','data':{'has_sawyer':hasattr(envs,'SawyerPushGoalEnv')},'timestamp':int(time.time()*1000)})+'\n')
    # #endregion
    print("✅ Using modern MuJoCo environments (no compilation needed!)")
except ImportError as e:
    # #region agent log
    import json,time;open('/Users/duruoli/A/A李杜若/1-科研/PhD/0/1-code/hdm2/.cursor/debug.log','a').write(json.dumps({'sessionId':'debug-session','runId':'pre-fix','hypothesisId':'H1,H2,H3','location':'__main__.py:18','message':'modern_envs import failed','data':{'error':str(e),'error_type':type(e).__name__},'timestamp':int(time.time()*1000)})+'\n')
    # #endregion
    print("⚠️  Falling back to old gcsl environments (requires mujoco_py)")
    from gcsl import envs
    from gcsl.envs.env_utils import DiscretizedActionEnv
from hdm.agent.core import DiscretePolicy, IndependentDiscretePolicy
from hdm.algo.core import Algo
from hdm.learn.core import Learner
from hdm.replay.core import Replay
from hdm.utils import mpi_utils
from hdm.utils import torch_utils
from hdm.utils import vec_env
from hdm.utils.run_utils import Monitor

ACTION_GRANULARITY = 3


def discretize_environment(env, env_params):
    if isinstance(env.action_space, Discrete):
        return env
    granularity = env_params.get('action_granularity', ACTION_GRANULARITY)
    env_discretized = DiscretizedActionEnv(env, granularity=granularity)
    return env_discretized


def default_markov_policy(env, env_params):
    policy_class = DiscretePolicy
    assert isinstance(env.action_space, Discrete)
    if env.action_space.n > 100:
        env_params['independent_policy'] = True
    
    if env_params['independent_policy']:
        print("Initializing IndependentDiscretePolicy ...")
        policy_class = IndependentDiscretePolicy
    
    env_params['action_space.n'] = env.action_space.n
    if hasattr(env.action_space, 'n_dims'):
        env_params['action_space.n_dims'] = env.action_space.n_dims
    env_params['action_space.granularity'] = ACTION_GRANULARITY
    return policy_class(
        env_params,
        layers=(400, 300),
        activ="relu",
    )


def get_env_and_agent(env, env_params):
    env = discretize_environment(env, env_params)
    policy = default_markov_policy(env, env_params)
    return env, policy


def get_env_with_id(num_envs, env_id):
    vec_fn = vec_env.SubprocVecEnv
    return vec_fn([lambda: gym.make(env_id) for _ in range(num_envs)])


def get_env_with_fn(num_envs, env_fn, *args, **kwargs):
    vec_fn = vec_env.SubprocVecEnv
    return vec_fn([lambda: env_fn(*args, **kwargs) for _ in range(num_envs)])


def get_env_alone(env_name):
    env = envs.create_env(env_name)
    env_params = envs.get_env_params(env_name)
    return discretize_environment(env, env_params)


def launch(args):
    env_name = args.env_name
    env = envs.create_env(env_name)
    env_params = envs.get_env_params(env_name)
    print(env_params)
    env_params.update(
        {'obs_dim': env.observation_space.shape[0],
         'goal_dim': env.goal_space.shape[0],
         'independent_policy': args.independent_policy}
    )
    env, agent = get_env_and_agent(env, env_params)
    
    rank = mpi_utils.get_rank()
    seed = args.seed + rank * args.n_workers
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_utils.use_cuda:
        torch.cuda.manual_seed(seed)
    
    monitor = Monitor()
    
    if args.n_workers > 1:
        env = vec_env.SubprocVecEnv([lambda: get_env_alone(args.env_name) for _ in range(args.n_workers)])
        env.seed(seed)
    
    ckpt_name = args.ckpt_name
    if len(ckpt_name) == 0:
        data_time = time.ctime().split()[1:4]
        ckpt_name = data_time[0] + '-' + data_time[1]
        time_list = np.array([float(i) for i in data_time[2].split(':')], dtype=np.float32)
        if mpi_utils.use_mpi():
            time_list = mpi_utils.bcast(time_list)
        for time_ in time_list:
            ckpt_name += '-' + str(int(time_))
        args.ckpt_name = ckpt_name
    
    replay = Replay(env, env_params, args)
    learner = Learner(agent, monitor, args)
    algo = Algo(
        env=env, env_params=env_params, args=args,
        agent=agent, replay=replay, monitor=monitor, learner=learner,
    )
    return algo


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--env_name', type=str, default='pointmass_rooms')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='experiments/')
    parser.add_argument('--ckpt_name', type=str, default='')
    parser.add_argument('--resume_ckpt', type=str, default='')
    
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--num_rollouts_per_mpi', type=int, default=1)
    
    parser.add_argument('--n_cycles', type=int, default=40)
    parser.add_argument('--optimize_every', type=int, default=50)
    parser.add_argument('--n_batches', type=int, default=50)
    parser.add_argument('--target_update_freq', type=int, default=10)
    
    parser.add_argument('--greedy_action', action='store_true')
    parser.add_argument('--random_act_prob', type=float, default=0.0)
    
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--future_p', type=float, default=0.85)
    parser.add_argument('--next_state_p', type=float, default=0.0)
    parser.add_argument('--relabeled_reward_only', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    
    parser.add_argument('--lr_actor', type=float, default=5.e-4)
    parser.add_argument('--start_policy_timesteps', type=int, default=1000)
    
    parser.add_argument('--n_initial_rollouts', type=int, default=200)
    parser.add_argument('--n_test_rollouts', type=int, default=50)
    
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--polyak', type=float, default=0.995)
    
    parser.add_argument('--independent_policy', action='store_true')
    
    parser.add_argument('--use_dqn', action='store_true')
    parser.add_argument('--double_dqn', action='store_true')
    parser.add_argument('--backup_strategy', type=str,
                        choices=['q_max', 'q_softmax', 'q_eps_greedy', 'q_soft_kl'],
                        default='q_max')
    parser.add_argument('--backup_temp', type=float, default=1.0)
    parser.add_argument('--backup_epsilon', type=float, default=0.1)
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--reward_bias', type=float, default=-1.0)
    parser.add_argument('--targ_clip', action='store_true')
    
    parser.add_argument('--hdm_gamma', type=float, default=0.85)
    parser.add_argument('--hdm_weights_min', type=float, default=-5.0)
    parser.add_argument('--hdm_weights_max', type=float, default=5.0)
    parser.add_argument('--hdm_q_coef', type=float, default=0.0)
    parser.add_argument('--hdm_q_normalizer', action='store_true')
    parser.add_argument('--hdm_online_o2', action='store_true')
    parser.add_argument('--hdm_backup_strategy', type=str,
                        choices=['q_max', 'q_softmax', 'q_eps_greedy', 'q_soft_kl', 'act_2'],
                        default='act_2')
    parser.add_argument('--hdm_bc', action='store_true')
    parser.add_argument('--hdm_weights_to_indicator', action='store_true')
    parser.add_argument('--hdm_gamma_use_auto', action='store_true')
    parser.add_argument('--hdm_weights_relabel_mask', action='store_true')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    n_threads = str(args.n_workers)
    if args.n_workers < 12:
        n_threads = str(12)
    
    os.environ['OMP_NUM_THREADS'] = n_threads
    os.environ['MKL_NUM_THREADS'] = n_threads
    os.environ['IN_MPI'] = n_threads
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    algo = launch(args)
    algo.run()
