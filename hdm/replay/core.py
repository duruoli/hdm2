import os.path as osp
import threading

import numpy as np
import torch

from hdm.utils import mpi_utils


def sample_her_transitions(buffer, batch_size, future_p=1.0, next_state_p=0.0):
    assert all(k in buffer for k in ['state', 'bg_state', 'act', 'ag_changed'])
    buffer['state_2'] = buffer['state'][:, 1:, :]
    buffer['act_2'] = buffer['act'][:, 1:]
    
    n_trajs = buffer['bg_state'].shape[0]
    horizon = buffer['bg_state'].shape[1]
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}
    
    her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
    
    future_offset = (np.random.uniform(size=batch_size) * (horizon - t_samples)).astype(int)
    if next_state_p > 0:
        future_offset[np.where(np.random.uniform(size=batch_size) < next_state_p)] = 0
    
    batch['r_relabel'] = np.zeros(batch_size)
    batch['r_relabel'][her_indexes] = (future_offset[her_indexes] == 0).astype(float)
    
    batch['relabel_mask'] = np.zeros(batch_size)
    batch['relabel_mask'][her_indexes] = 1.0
    
    future_t = (t_samples + 1 + future_offset)
    batch['bg_state'][her_indexes] = buffer['state'][ep_idxes[her_indexes], future_t[her_indexes]]
    batch['future_state'] = buffer['state'][ep_idxes, future_t].copy()
    batch['offset'] = future_offset.copy()
    
    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(k in batch for k in [
        'state', 'bg_state', 'act', 'state_2', 'act_2', 'future_state', 'offset', 'ag_changed', 'r_relabel'])
    return batch


class Replay:
    def __init__(self, env, env_params, args, name='replay'):
        self.env = env
        self.env_params = env_params
        self.args = args
        self.goal_threshold = self.env_params['goal_threshold']
        
        self.horizon = env_params['max_trajectory_length']
        self.size = args.buffer_size // self.horizon
        
        if mpi_utils.is_root():
            print('replay_size:', self.size)
        
        self.current_size = 0
        self.n_transitions_stored = 0
        
        self.buffers = dict(state=np.zeros((self.size, self.horizon + 1, env.state_space.shape[0])),
                            bg_state=np.zeros((self.size, self.horizon, env.state_space.shape[0])),
                            act=np.zeros((self.size, self.horizon + 1)),
                            ag_changed=np.zeros((self.size, self.horizon + 1)))
        
        self.lock = threading.Lock()
        self._save_file = str(name) + '_' + str(mpi_utils.get_rank()) + '.pt'
    
    def store(self, episodes):
        state = episodes['state']
        bg_state = episodes['bg_state']
        act = episodes['act']
        ag_changed = episodes['ag_changed']
        batch_size = state.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(batch_size=batch_size)
            self.buffers['state'][idxs] = state.copy()
            self.buffers['bg_state'][idxs] = bg_state.copy()
            self.buffers['act'][idxs] = act.copy()
            self.buffers['ag_changed'][idxs] = ag_changed.copy()
            self.n_transitions_stored += self.horizon * batch_size
    
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        transitions = sample_her_transitions(
            temp_buffers, batch_size, future_p=self.args.future_p, next_state_p=self.args.next_state_p)
        state = transitions['state']
        state_2 = transitions['state_2']
        bg_state = transitions['bg_state']
        future_state = transitions['future_state']
        # reward is calculated between ag2 and bg
        if self.args.relabeled_reward_only:
            reward = transitions['r_relabel']
        else:
            reward = (self.env.goal_distance(state_2, bg_state) < self.goal_threshold).astype(float)
        batch = {
            'ob': self.env.observation(state),
            'ob_2': self.env.observation(state_2),
            'bg': self.env.extract_goal(bg_state),
            'future_ag': self.env.extract_goal(future_state),
            'act': transitions['act'],
            'act_2': transitions['act_2'],
            'offset': transitions['offset'],
            'reward': reward,
            'relabel_mask': transitions['relabel_mask'],
        }
        return batch
    
    def _get_storage_idx(self, batch_size):
        if self.current_size + batch_size <= self.size:
            idx = np.arange(self.current_size, self.current_size + batch_size)
        elif self.current_size < self.size:
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, batch_size - len(idx_a))
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, batch_size)
        self.current_size = min(self.size, self.current_size + batch_size)
        if batch_size == 1:
            idx = idx[0]
        return idx
    
    def state_dict(self):
        return dict(
            current_size=self.current_size,
            n_transitions_stored=self.n_transitions_stored,
            buffers=self.buffers,
        )
    
    def load_state_dict(self, state_dict):
        self.current_size = state_dict['current_size']
        self.n_transitions_stored = state_dict['n_transitions_stored']
        self.buffers = state_dict['buffers']
    
    def save(self, path):
        state_dict = self.state_dict()
        save_path = osp.join(path, self._save_file)
        torch.save(state_dict, save_path)
    
    def load(self, path):
        load_path = osp.join(path, self._save_file)
        state_dict = torch.load(load_path)
        self.load_state_dict(state_dict)
