import math
import os.path as osp

import numpy as np
import torch
import torch.nn as nn

from hdm.utils import mpi_utils, torch_utils, net_utils


def get_activ(activ_name):
    if activ_name == "relu":
        return nn.ReLU
    elif activ_name == "elu":
        return nn.ELU
    elif activ_name == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError


class Mlp(nn.Module):
    def __init__(self, dim_input, dim_output, layers=(256, 256), activ="relu"):
        super().__init__()
        net_layers = []
        dim = dim_input
        for hid_dim in layers:
            net_layers.append(nn.Linear(dim, hid_dim))
            net_layers.append(get_activ(activ)(inplace=True))
            dim = hid_dim
        net_layers.append(nn.Linear(dim, dim_output))
        self.net = nn.Sequential(*net_layers)
    
    def forward(self, x):
        return self.net(x)


class StateGoalNet(nn.Module):
    def __init__(self, env_params, dim_output=1, layers=(512, 512), activ="relu"):
        super().__init__()
        input_dim = env_params['obs_dim'] + env_params['goal_dim']
        self.net = Mlp(input_dim, dim_output, layers=layers, activ=activ)
    
    def forward(self, state, goal):
        embed = torch.cat([state, goal], dim=-1)
        return self.net(embed)


def logits_selection(logits, target):
    assert logits.dim() == 2
    assert not target.requires_grad
    if target.dim() == 2:
        target = target.squeeze(1)
    assert target.dim() == 1
    return logits.gather(1, target.long().unsqueeze(1)).squeeze(1)


def bc_loss_one_dimensional(logits, target):
    logits_select = logits_selection(logits, target)
    logits_logsumexp = torch.logsumexp(logits, dim=1)
    bc_loss = - (logits_select - logits_logsumexp)
    return bc_loss, dict(logits_select=logits_select, logits_logsumexp=logits_logsumexp)


def q_target_selection(logits, temp=1.0, backup_epsilon=0.1):
    q_max_idx = torch.argmax(logits, dim=-1)
    q_max = logits_selection(logits, q_max_idx)
    action_dim = logits.size(-1)
    q_soft_kl = temp * (torch.logsumexp(logits / temp, dim=1) - math.log(action_dim))
    q_softmax = (torch.softmax(logits / temp, dim=1) * logits).sum(dim=1)
    one_hot_action = nn.functional.one_hot(q_max_idx, action_dim).float()
    epsilon_action = torch.ones_like(logits) / action_dim
    eps_greedy_action = (1 - backup_epsilon) * one_hot_action + backup_epsilon * epsilon_action
    q_eps_greedy = (eps_greedy_action * logits).sum(dim=1)
    return dict(q_max=q_max, q_softmax=q_softmax, q_eps_greedy=q_eps_greedy, q_soft_kl=q_soft_kl)


class DiscretePolicy:
    def __init__(self, env_params, layers=(512, 512), activ="relu"):
        super().__init__()
        self.action_space_n = env_params['action_space.n']
        self.actor = StateGoalNet(
            env_params, dim_output=self.action_space_n, layers=layers, activ=activ
        )
        self.target_actor = StateGoalNet(
            env_params, dim_output=self.action_space_n, layers=layers, activ=activ
        )
        self.target_actor.load_state_dict(self.actor.state_dict())
        net_utils.set_requires_grad(self.target_actor, allow_grad=False)
        if torch_utils.use_cuda:
            self.actor.cuda()
            self.target_actor.cuda()
        self._save_file = 'policy.pt'
    
    def forward(self, obs, goal, target_network=False):
        obs = self.preprocess(obs)
        goal = self.preprocess(goal)
        network = self.target_actor if target_network else self.actor
        return network.forward(obs, goal)
    
    def preprocess(self, x, cast_to_2d=True):
        if type(x) == torch.Tensor:
            x_tensor = x
        else:
            x_tensor = torch.from_numpy(x).float().to(device=torch_utils.device)
        if x_tensor.ndim == 1 and cast_to_2d:
            x_tensor = x_tensor.reshape(1, -1)
        return x_tensor
    
    def act(self, obs, goal, greedy=False, random_act_prob=0.0):
        logits = self.forward(obs, goal)
        if greedy:
            samples = torch.argmax(logits, dim=-1)
        else:
            samples = torch.distributions.categorical.Categorical(logits=logits).sample()
        np_samples = samples.detach().cpu().numpy()
        if random_act_prob > 0:
            batch_size = len(np_samples)
            random_samples = np.random.choice(self.action_space_n, size=batch_size)
            random_idx = np.random.rand(batch_size) < random_act_prob
            np_samples = np.where(random_idx, random_samples, np_samples)
        return np_samples.squeeze()
    
    def bc_loss(self, obs, goal, actions):
        actions = self.preprocess(actions, cast_to_2d=False)
        logits = self.forward(obs, goal)
        bc_loss, bc_dict = bc_loss_one_dimensional(logits, actions)
        return bc_loss, bc_dict
    
    def q_function(self, obs, goal, actions, target_network=False, temp=1.0, backup_epsilon=0.0):
        actions = self.preprocess(actions, cast_to_2d=False)
        logits = self.forward(obs, goal, target_network=target_network)
        q_action = logits_selection(logits, actions)
        q_dict = q_target_selection(logits, temp=temp, backup_epsilon=backup_epsilon)
        return q_action, q_dict
    
    def target_update(self, polyak=0.995):
        net_utils.target_soft_update(source=self.actor, target=self.target_actor, polyak=polyak)
    
    def normalizer_update(self, obs, goal):
        pass
    
    def train(self):
        self.actor.train()
    
    def eval(self):
        self.actor.eval()
    
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
    
    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.target_actor.load_state_dict(state_dict['target_actor'])


class IndependentDiscretePolicy:
    def __init__(self, env_params, layers=(512, 512), activ="relu"):
        super().__init__()
        self.n_dims = env_params['action_space.n_dims']
        self.granularity = env_params['action_space.granularity']
        self.act_dim = self.n_dims * self.granularity
        self.action_space_n = env_params['action_space.n']
        self.actor = StateGoalNet(
            env_params, dim_output=self.act_dim, layers=layers, activ=activ
        )
        self.target_actor = StateGoalNet(
            env_params, dim_output=self.act_dim, layers=layers, activ=activ
        )
        self.target_actor.load_state_dict(self.actor.state_dict())
        net_utils.set_requires_grad(self.target_actor, allow_grad=False)
        if torch_utils.use_cuda:
            self.actor.cuda()
            self.target_actor.cuda()
        self._save_file = 'policy.pt'
    
    def flatten_action(self, action):
        assert action.ndim == 2 and action.size(1) == self.n_dims
        multipliers = self.granularity ** torch.arange(self.n_dims).to(device=torch_utils.device)
        flattened = (action * multipliers).sum(1)
        return flattened.int()
    
    def unflatten_action(self, action):
        digits = []
        output = action.float()
        for _ in range(self.n_dims):
            digits.append(output % self.granularity)
            # output = torch.div(output, self.granularity, rounding_mode='floor')
            # To make it compatible with different torch versions:
            output = torch.trunc(torch.true_divide(output, self.granularity))
        return torch.stack(digits, dim=-1)
    
    def forward(self, obs, goal, target_network=False):
        obs = self.preprocess(obs)
        goal = self.preprocess(goal)
        network = self.target_actor if target_network else self.actor
        return network.forward(obs, goal)
    
    def preprocess(self, x, cast_to_2d=True):
        if type(x) == torch.Tensor:
            x_tensor = x
        else:
            x_tensor = torch.from_numpy(x).float().to(device=torch_utils.device)
        if x_tensor.ndim == 1 and cast_to_2d:
            x_tensor = x_tensor.reshape(1, -1)
        return x_tensor
    
    def act(self, obs, goal, greedy=False, random_act_prob=0.0):
        logits = self.forward(obs, goal)
        logits = logits.view(-1, self.n_dims, self.granularity)
        if greedy:
            samples = torch.argmax(logits, dim=-1)
        else:
            samples = torch.distributions.categorical.Categorical(logits=logits).sample()
        samples = self.flatten_action(samples)
        np_samples = samples.detach().cpu().numpy()
        if random_act_prob > 0:
            batch_size = len(np_samples)
            random_samples = np.random.choice(self.action_space_n, size=batch_size)
            random_idx = np.random.rand(batch_size) < random_act_prob
            np_samples = np.where(random_idx, random_samples, np_samples)
        return np_samples.squeeze()
    
    def bc_loss(self, obs, goal, actions):
        actions = self.preprocess(actions, cast_to_2d=False)
        batch_size = actions.size(0)
        
        actions_per_dim = self.unflatten_action(actions)
        actions_per_dim = actions_per_dim.view(batch_size * self.n_dims)
        
        logits = self.forward(obs, goal)
        logits_per_dim = logits.view(batch_size * self.n_dims, self.granularity)
        
        assert actions_per_dim.size(0) == logits_per_dim.size(0)
        loss_per_dim, bc_dict = bc_loss_one_dimensional(logits_per_dim, actions_per_dim)
        
        loss = loss_per_dim.reshape(batch_size, self.n_dims)
        return loss.sum(1), bc_dict
    
    def q_function(self, obs, goal, actions, target_network=False, temp=1.0, backup_epsilon=0.0):
        actions = self.preprocess(actions, cast_to_2d=False)
        batch_size = actions.size(0)
        
        actions_per_dim = self.unflatten_action(actions)
        actions_per_dim = actions_per_dim.view(batch_size * self.n_dims)
        
        logits = self.forward(obs, goal, target_network=target_network)
        logits_per_dim = logits.view(batch_size * self.n_dims, self.granularity)
        
        assert actions_per_dim.size(0) == logits_per_dim.size(0)
        q_action_per_dim = logits_selection(logits_per_dim, actions_per_dim)
        q_action_per_dim = q_action_per_dim.reshape(batch_size, self.n_dims)
        q_action = torch.sum(q_action_per_dim, 1)
        
        q_dict_per_dim = q_target_selection(logits_per_dim, temp=temp, backup_epsilon=backup_epsilon)
        q_dict = {
            q_key: q_tensor.reshape(batch_size, self.n_dims).sum(1) for q_key, q_tensor in q_dict_per_dim.items()
        }
        return q_action, q_dict
    
    def target_update(self, polyak=0.995):
        net_utils.target_soft_update(source=self.actor, target=self.target_actor, polyak=polyak)
    
    def normalizer_update(self, obs, goal):
        pass
    
    def train(self):
        self.actor.train()
    
    def eval(self):
        self.actor.eval()
    
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
    
    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.target_actor.load_state_dict(state_dict['target_actor'])
