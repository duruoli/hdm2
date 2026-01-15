import os
import os.path as osp
import sys

import numpy as np
import torch

from hdm import logger
from hdm.utils import mpi_utils
from hdm.utils.run_utils import Timer, log_config, merge_configs


class BaseAlgo:
    def __init__(
        self,
        env, env_params, args,
        agent, replay, monitor, learner,
        name='algo',
    ):
        self.env = env
        self.env_params = env_params
        self.args = args
        
        assert hasattr(self.env.action_space, 'n')
        self.goal_threshold = self.env_params['goal_threshold']
        
        self.agent = agent
        self.replay = replay
        self.monitor = monitor
        self.learner = learner
        
        self.timer = Timer()
        self.start_time = self.timer.current_time
        self.total_timesteps = 0
        
        self.env_steps = 0
        self.opt_steps = 0
        self.best_success_rate = 0.0
        
        self.num_envs = 1
        if hasattr(self.env, 'num_envs'):
            self.num_envs = getattr(self.env, 'num_envs')
        
        self.n_mpi = mpi_utils.get_size()
        self._save_file = str(name) + '.pt'
        
        if len(args.resume_ckpt) > 0:
            resume_path = osp.join(
                osp.join(self.args.save_dir, self.args.env_name),
                osp.join(args.resume_ckpt, 'state'))
            self.load_all(resume_path)
        
        self.log_path = osp.join(osp.join(self.args.save_dir, self.args.env_name), args.ckpt_name)
        self.model_path = osp.join(self.log_path, 'state')
        if mpi_utils.is_root():
            os.makedirs(self.model_path, exist_ok=True)
            logger.configure(dir=self.log_path, format_strs=["csv", "stdout"])
            config_list = [env_params.copy(), args.__dict__.copy(), {'NUM_MPI': mpi_utils.get_size()}]
            log_config(config=merge_configs(config_list), output_dir=self.log_path)
    
    def get_actions(self, ob, bg, greedy=False, random_act_prob=0.0):
        return self.agent.act(ob, bg, greedy=greedy, random_act_prob=random_act_prob)
    
    def run_eval(self, use_test_env=False):
        self.agent.eval()
        env = self.env
        if use_test_env and hasattr(self, 'test_env'):
            env = self.test_env
        total_success_count = 0
        total_trial_count = 0
        dist_list = []
        
        self.eval_count += 1
        log_this_eval = self.debug_log_trajectories
        
        for n_test in range(self.args.n_test_rollouts):
            state = env.reset()
            state_origin = state.copy()
            ob = env.observation(state)
            goal_state = env.sample_goal()
            bg = env.extract_goal(goal_state)
            ag_changed = None
            
            # Log ALL rollouts in eval (changed from first only)
            log_this_rollout = log_this_eval
            traj_data = [] if log_this_rollout else None
            has_alpha_zero = False  # Track if alpha=0 occurs in this rollout
            
            for timestep in range(self.env_params['max_trajectory_length']):
                act = self.get_actions(ob, bg, greedy=True, random_act_prob=0.0)
                
                # Log trajectory data
                if log_this_rollout:
                    gripper_pos = ob[0:3] if ob.ndim == 1 else ob[0, 0:3]
                    object_pos = ob[4:7] if ob.ndim == 1 else ob[0, 4:7]
                    goal_object_pos = bg[3:6] if bg.ndim == 1 else bg[0, 3:6]
                    
                    gripper_to_obj_dist = float(np.linalg.norm(gripper_pos - object_pos))
                    reach_threshold = 0.05  # Same as in metaworld_wrapper.py
                    alpha = 0.0 if gripper_to_obj_dist < reach_threshold else 1.0
                    
                    if alpha == 0.0:
                        has_alpha_zero = True
                    
                    traj_data.append({
                        'timestep': timestep,
                        'gripper_pos': tuple(gripper_pos.tolist()),
                        'object_pos': tuple(object_pos.tolist()),
                        'goal_obj_pos': tuple(goal_object_pos.tolist()),
                        'action': int(act) if act.ndim == 0 else int(act[0]),
                        'gripper_to_obj_dist': gripper_to_obj_dist,
                        'obj_to_goal_dist': float(np.linalg.norm(object_pos - goal_object_pos)),
                        'alpha': alpha,
                    })
                
                state, _, _, _ = env.step(act)
                ob = env.observation(state)
                ag_changed = self.env.goal_distance(state, state_origin) > self.goal_threshold
                self.monitor.store(Inner_Test_AgChangeRatio=np.mean(ag_changed))
            
            # Save eval trajectory log
            if log_this_rollout and traj_data:
                import pandas as pd
                df = pd.DataFrame(traj_data)
                self.trajectory_logs.append({
                    'type': 'eval',
                    'eval_num': self.eval_count,
                    'rollout_num': n_test,
                    'env_steps': self.env_steps,
                    'has_alpha_zero': has_alpha_zero,
                    'data': df
                })
            
            assert ag_changed is not None
            self.monitor.store(TestAgChangeRatio=np.mean(ag_changed))
            if self.num_envs > 1:
                final_dist = env.goal_distance(state, goal_state)
                success_vec = (final_dist < self.goal_threshold)
            else:
                final_dist = env.goal_distance(state[None], goal_state[None])
                success_vec = (final_dist < self.goal_threshold)
            assert final_dist.shape[0] == self.num_envs
            total_trial_count += final_dist.shape[0]
            total_success_count += np.sum(success_vec)
            dist_list.append(final_dist.copy())
        success_rate = total_success_count / total_trial_count
        avg_final_dist = np.mean(np.concatenate(dist_list, axis=0))
        if mpi_utils.use_mpi():
            success_rate = mpi_utils.global_mean(np.array([success_rate]))[0]
            avg_final_dist = mpi_utils.global_mean(np.array([avg_final_dist]))[0]
        return success_rate, avg_final_dist
    
    def log_everything(self):
        for log_name in self.monitor.epoch_dict:
            log_item = self.monitor.log(log_name)
            if mpi_utils.use_mpi():
                log_item_k = log_item.keys()
                log_item_v = np.array(list(log_item.values()))
                log_item_v = mpi_utils.global_mean(log_item_v)
                log_item = dict(zip(log_item_k, log_item_v))
            logger.record_tabular(log_name, log_item['mean'])
        logger.record_tabular('TotalTimeSteps', self.total_timesteps)
        logger.record_tabular('Time', self.timer.current_time - self.start_time)
        if mpi_utils.is_root():
            logger.dump_tabular()
    
    def state_dict(self):
        raise NotImplementedError
    
    def load_state_dict(self, state_dict):
        raise NotImplementedError
    
    def save(self, path):
        if mpi_utils.is_root():
            state_dict = self.state_dict()
            save_path = osp.join(path, self._save_file)
            torch.save(state_dict, save_path)
    
    def load(self, path):
        load_path = osp.join(path, self._save_file)
        try:
            state_dict = torch.load(load_path)
        except RuntimeError:
            state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)
    
    def save_all(self, path):
        self.save(path)
        self.agent.save(path)
        self.replay.save(path)
        self.learner.save(path)
    
    def load_all(self, path):
        self.load(path)
        self.agent.load(path)
        self.replay.load(path)
        self.learner.load(path)


class Algo(BaseAlgo):
    def __init__(
        self,
        env, env_params, args,
        agent, replay, monitor, learner,
        name='algo',
    ):
        super().__init__(
            env, env_params, args,
            agent, replay, monitor, learner,
            name=name,
        )
        
        # Trajectory logging for debugging
        self.debug_log_trajectories = getattr(args, 'debug_log_trajectories', False)
        self.collect_count = 0
        self.eval_count = 0
        self.trajectory_logs = []
    
    def agent_optimize(self):
        self.timer.start('train')
        
        self.agent.train()
        for n_train in range(self.args.n_batches):
            batch = self.replay.sample(batch_size=self.args.batch_size)
            self.learner.update(batch)
            self.opt_steps += 1
            if self.opt_steps % self.args.target_update_freq == 0:
                self.learner.target_update()
        self.agent.eval()
        
        self.timer.end('train')
        self.monitor.store(TimePerTrainIter=self.timer.get_time('train') / self.args.n_batches)
    
    def collect_experience(self, greedy=False, random_act_prob=0.0, train_agent=True):
        state_list, bg_state_list, act_list = [], [], []
        state = self.env.reset()
        ob = self.env.observation(state)
        goal_state = self.env.sample_goal()
        bg = self.env.extract_goal(goal_state)
        
        state_origin = state.copy()
        ag_changed = None
        self.agent.eval()
        
        # # Track if this is the first trajectory in this collect period
        # self.collect_count += 1
        # log_this_traj = self.debug_log_trajectories and (self.collect_count == 1 or 
        #                 (self.collect_count - 1) % self.args.num_rollouts_per_mpi == 0)
        
        # traj_data = [] if log_this_traj else None
        
        for timestep in range(self.env_params['max_trajectory_length']):
            act = self.get_actions(ob, bg, greedy=greedy, random_act_prob=random_act_prob)
            
            # # Log trajectory data for debugging
            # if log_this_traj:
            #     # Extract gripper and object positions from observation
            #     gripper_pos = ob[0:3] if ob.ndim == 1 else ob[0, 0:3]
            #     object_pos = ob[4:7] if ob.ndim == 1 else ob[0, 4:7]
            #     goal_object_pos = bg[3:6] if bg.ndim == 1 else bg[0, 3:6]  # Goal object pos from achieved_goal
                
            #     traj_data.append({
            #         'timestep': timestep,
            #         'gripper_pos': tuple(gripper_pos.tolist()),
            #         'object_pos': tuple(object_pos.tolist()),
            #         'goal_obj_pos': tuple(goal_object_pos.tolist()),
            #         'action': int(act) if act.ndim == 0 else int(act[0]),
            #         'gripper_to_obj_dist': float(np.linalg.norm(gripper_pos - object_pos)),
            #         'obj_to_goal_dist': float(np.linalg.norm(object_pos - goal_object_pos)),
            #     })
            
            state_list.append(state.copy())
            bg_state_list.append(goal_state.copy())
            act_list.append(act.copy())
            state, _, _, _ = self.env.step(act)
            ob = self.env.observation(state)
            ag_changed = self.env.goal_distance(state, state_origin) > self.goal_threshold
            self.monitor.store(Inner_Train_AgChangeRatio=np.mean(ag_changed))
            self.total_timesteps += self.num_envs * self.n_mpi
            for every_env_step in range(self.num_envs): #number of parallel envs
                self.env_steps += 1
                if self.env_steps % self.args.optimize_every == 0 \
                    and self.env_steps > self.args.start_policy_timesteps \
                    and train_agent:
                    self.agent_optimize()
        
        # # Save trajectory log
        # if log_this_traj and traj_data:
        #     import pandas as pd
        #     df = pd.DataFrame(traj_data)
        #     self.trajectory_logs.append({
        #         'type': 'train',
        #         'collect_num': self.collect_count,
        #         'env_steps': self.env_steps,
        #         'data': df
        #     })
        
        state_list.append(state.copy())
        act = self.get_actions(ob, bg, greedy=greedy, random_act_prob=random_act_prob)
        act_list.append(act.copy())
        ag_changed_list = [ag_changed for _ in range(len(act_list))]
        
        experience = {'state': state_list, 'bg_state': bg_state_list, 'act': act_list, 'ag_changed': ag_changed_list}
        experience = {k: np.array(v) for k, v in experience.items()}
        if experience['state'].ndim == 2:
            experience = {k: np.expand_dims(v, 0) for k, v in experience.items()}
        else:
            experience = {k: np.swapaxes(v, 0, 1) for k, v in experience.items()}
        
        bg_achieve = self.env.goal_distance(state, goal_state) < self.goal_threshold
        self.monitor.store(TrainSuccess=np.mean(bg_achieve))
        assert ag_changed is not None
        self.monitor.store(TrainAgChangeRatio=np.mean(ag_changed))
        self.replay.store(experience)
        self.update_normalizer(experience)
    
    def update_normalizer(self, buffer):
        pass
    
    def save_trajectory_logs(self):
        """Save all logged trajectories to CSV files."""
        if self.debug_log_trajectories and mpi_utils.is_root():
            import pandas as pd
            log_dir = osp.join(self.log_path, 'trajectory_logs')
            os.makedirs(log_dir, exist_ok=True)
            
            for i, log_entry in enumerate(self.trajectory_logs):
                if log_entry['type'] == 'train':
                    filename = f"train_{log_entry['collect_num']:04d}_steps{log_entry['env_steps']}.csv"
                else:
                    # Eval filename with rollout number and alpha marker
                    alpha_marker = "_alpha0" if log_entry.get('has_alpha_zero', False) else ""
                    filename = f"eval_epoch{log_entry['eval_num']:04d}_rollout{log_entry['rollout_num']:02d}_steps{log_entry['env_steps']}{alpha_marker}.csv"
                filepath = osp.join(log_dir, filename)
                log_entry['data'].to_csv(filepath, index=False)
                print(f"Saved trajectory log: {filepath}")
            
            print(f"\nTotal trajectory logs saved: {len(self.trajectory_logs)}")
    
    def run(self): #epoch = evaluation (multiple test rollouts) > cycle > rollout= trajectory, episode > policy update
        for n_init_rollout in range(self.args.n_initial_rollouts // self.num_envs):
            self.collect_experience(greedy=False, random_act_prob=1.0, train_agent=False) #interact with env and store trajectories in replay buffer; trigger learning (agent_optimize())
        
        epoch = 0
        max_epochs = self.args.max_epochs if self.args.max_epochs is not None else float('inf')
        while self.env_steps < self.env_params['max_timesteps'] and epoch < max_epochs: #Stop when either max_timesteps or max_epochs is reached
            epoch += 1
            
            success_rate, final_dist = self.run_eval() #evaluate at the start of each epoch (test previous updated policy)
            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate
            if mpi_utils.is_root():
                print('Epoch %d eval | final dist %.3f | success rate %.3f | best success %.3f' % (
                epoch, final_dist, success_rate, self.best_success_rate))
                print('ckpt_name:', self.args.ckpt_name)
            logger.record_tabular("Epoch", epoch)
            logger.record_tabular('TestSuccessRate', success_rate)
            logger.record_tabular('TestFinalDist', final_dist)
            logger.record_tabular('TestBestSuccessRate', self.best_success_rate)
            
            if mpi_utils.is_root():
                print('Epoch %d: Iter (out of %d)=' % (epoch, self.args.n_cycles), end=' ')
                sys.stdout.flush()
            
            for n_iter in range(self.args.n_cycles):
                if mpi_utils.is_root():
                    print("%d" % n_iter, end=' ' if n_iter < self.args.n_cycles - 1 else '\n')
                    sys.stdout.flush()
                self.timer.start('rollout')
                
                for n_rollout in range(self.args.num_rollouts_per_mpi):
                    self.collect_experience(
                        greedy=self.args.greedy_action,
                        random_act_prob=self.args.random_act_prob,
                        train_agent=True,
                    )
                
                self.timer.end('rollout')
                self.monitor.store(TimePerSeqRollout=self.timer.get_time('rollout') / self.args.num_rollouts_per_mpi)
            
            self.monitor.store(env_steps=self.env_steps)
            self.monitor.store(opt_steps=self.opt_steps)
            self.monitor.store(replay_size=self.replay.current_size)
            self.monitor.store(replay_fill_ratio=float(self.replay.current_size / self.replay.size))
            
            self.log_everything()
            
            self.save_all(self.model_path)
        
        # Print why training stopped
        if mpi_utils.is_root():
            if epoch >= max_epochs:
                print(f'\n{"="*50}')
                print(f'Training stopped: reached max_epochs={int(max_epochs)}')
                print(f'Final: epoch={epoch}, env_steps={self.env_steps}')
                print(f'{"="*50}\n')
            else:
                print(f'\n{"="*50}')
                print(f'Training stopped: reached max_timesteps={self.env_params["max_timesteps"]}')
                print(f'Final: epoch={epoch}, env_steps={self.env_steps}')
                print(f'{"="*50}\n')
        
        # Save trajectory logs at the end of training
        self.save_trajectory_logs()
    
    def state_dict(self):
        return dict(total_timesteps=self.total_timesteps)
    
    def load_state_dict(self, state_dict):
        self.total_timesteps = state_dict['total_timesteps']
