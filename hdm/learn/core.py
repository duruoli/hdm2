import os.path as osp

import numpy as np
import torch
from torch.optim import Adam

from hdm.utils import mpi_utils, torch_utils, nan_police


def to_numpy(x):
    return x.detach().float().cpu().numpy()


def dict_to_numpy(tensor_dict):
    return {
        k: to_numpy(v) for k, v in tensor_dict.items()
    }


def torch_mean_to_numpy(tensor_dict):
    return {
        k: v.mean().item() for k, v in tensor_dict.items()
    }


class Learner:
    def __init__(
        self,
        agent,
        monitor,
        args,
        name='learner',
    ):
        self.agent = agent
        self.monitor = monitor
        self.args = args
        
        assert hasattr(agent, 'actor')
        self.pi_optim = Adam(agent.actor.parameters(), lr=args.lr_actor)
        
        self._save_file = str(name) + '.pt'
    
    def actor_loss(self, batch):
        ob, act, bg = batch['ob'], batch['act'], batch['bg']
        future_ag = batch['future_ag']
        loss_bc, bc_dict = self.agent.bc_loss(ob, future_ag, act)
        loss_bc = loss_bc.mean()
        self.monitor.store(
            Loss_bc=loss_bc.item(),
        )
        monitor_log = dict(
            q_values=bc_dict['logits_select'],
            Z_values=bc_dict['logits_logsumexp'],
        )
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_bc
    
    def dqn_loss(self, batch):
        ob, act, bg = batch['ob'], batch['act'], batch['bg']
        ob_2, act_2 = batch['ob_2'], batch['act_2']
        offset = torch_utils.from_numpy(batch['offset'])
        reward = batch['reward']
        reward = reward * self.args.reward_scale + self.args.reward_bias
        reward = torch_utils.from_numpy(reward)
        relabel_mask = torch_utils.from_numpy(batch['relabel_mask'])
        
        with torch.no_grad():
            if self.args.double_dqn:
                act_2 = torch.argmax(self.agent.forward(ob_2, bg), dim=-1)
            q_o2_a2, q_o2_dict = self.agent.q_function(
                ob_2, bg, act_2, target_network=True,
                temp=self.args.backup_temp, backup_epsilon=self.args.backup_epsilon,
            )
            q_o2_targ = q_o2_dict[self.args.backup_strategy]
            if self.args.double_dqn:
                q_o2_targ = q_o2_a2
            q_targ = reward + self.args.gamma * q_o2_targ
            if self.args.targ_clip:
                q_targ = q_targ.clamp(max=0.0)
        
        q_bg, q_bg_dict = self.agent.q_function(ob, bg, act, target_network=False, temp=self.args.backup_temp)
        loss_her = (q_bg - q_targ).pow(2).mean()
        
        hdm_gamma_auto = offset.mean() / (1 + offset.mean())
        
        with torch.no_grad():
            hdm_q_o1 = q_bg
            hdm_q_o2 = q_o2_a2
            hdm_q_o2_dict = q_o2_dict
            if self.args.hdm_online_o2:
                hdm_q_o2, hdm_q_o2_dict = self.agent.q_function(
                    ob_2, bg, act_2, target_network=False,
                    temp=self.args.backup_temp, backup_epsilon=self.args.backup_epsilon,
                )
            if self.args.hdm_backup_strategy != 'act_2':
                hdm_q_o2 = hdm_q_o2_dict[self.args.hdm_backup_strategy]
            hdm_gamma = self.args.hdm_gamma
            if self.args.hdm_gamma_use_auto:
                hdm_gamma = hdm_gamma_auto.item()
            hdm_gamma_zero_point = torch.exp(hdm_q_o1).mean() / torch.exp(hdm_q_o2).mean().clamp(min=0.01)
            hdm_weights = torch.exp(hdm_q_o1) - hdm_gamma * torch.exp(hdm_q_o2)
            hdm_weights = hdm_weights.clamp(min=self.args.hdm_weights_min, max=self.args.hdm_weights_max)
            if self.args.hdm_weights_to_indicator:
                hdm_weights = - (hdm_weights < 0).float()
            if self.args.hdm_weights_relabel_mask:
                hdm_weights *= relabel_mask
        
        hdm_q_to_minimize = q_bg
        if self.args.hdm_bc:
            hdm_q_to_minimize = q_bg - q_bg_dict['q_soft_kl']
        loss_hdm_q = (hdm_weights * hdm_q_to_minimize).mean()
        
        if self.args.hdm_q_normalizer:
            loss_hdm_q *= 1.0 / hdm_q_to_minimize.abs().mean().detach().clamp(min=1.0)
        
        loss_q = loss_her + self.args.hdm_q_coef * loss_hdm_q
        
        self.monitor.store(
            Loss_her=loss_her.item(),
            Loss_hdm=loss_hdm_q.item(),
            Loss_q=loss_q.item(),
            hdm_gamma_zero_point=hdm_gamma_zero_point.item(),
            offset=offset.mean().item(),
            hdm_gamma_auto=hdm_gamma_auto.item(),
            hdm_gamma=hdm_gamma,
            log_hdm_gamma=np.log(hdm_gamma),
            exp_avg_reward=torch.exp(reward.mean()).item(),
            exp_reward_avg=torch.exp(reward).mean().item(),
        )
        monitor_log = dict(
            q_targ=q_targ,
            q_bg=q_bg,
            q_o2_a2=q_o2_a2,
            q_o2_targ=q_o2_targ,
            q_o2_max=q_o2_dict['q_max'],
            q_o2_soft=q_o2_dict['q_softmax'],
            q_o2_soft_kl=q_o2_dict['q_soft_kl'],
            q_o2_eps_greedy=q_o2_dict['q_eps_greedy'],
            reward=reward,
            hdm_weights=hdm_weights,
            hdm_q_o1=hdm_q_o1,
            hdm_q_o2=hdm_q_o2,
            hdm_q_to_minimize=hdm_q_to_minimize,
        )
        self.monitor.store(**torch_mean_to_numpy(monitor_log))
        
        return loss_q
    
    def update(self, batch):
        if self.args.use_dqn:
            loss_actor = self.dqn_loss(batch)
        else:
            loss_actor = self.actor_loss(batch)
        self.pi_optim.zero_grad()
        loss_actor.backward()
        if nan_police.grad_has_nan(self.agent.actor.parameters()):
            import pdb
            pdb.set_trace()
        if mpi_utils.use_mpi():
            mpi_utils.sync_grads(self.agent.actor, scale_grad_by_procs=True)
        self.pi_optim.step()
    
    def target_update(self):
        self.agent.target_update(polyak=self.args.polyak)
    
    def state_dict(self):
        return dict(
            pi_optim=self.pi_optim.state_dict(),
        )
    
    def load_state_dict(self, state_dict):
        self.pi_optim.load_state_dict(state_dict['pi_optim'])
    
    def save(self, path):
        if mpi_utils.is_root():
            state_dict = self.state_dict()
            save_path = osp.join(path, self._save_file)
            torch.save(state_dict, save_path)
    
    def load(self, path):
        load_path = osp.join(path, self._save_file)
        state_dict = torch.load(load_path)
        self.load_state_dict(state_dict)
