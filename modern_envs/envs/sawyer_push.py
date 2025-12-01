"""
Modern SawyerPush environment using mujoco (not mujoco_py).

This is a port of the original GCSL SawyerPush environment to use
modern MuJoCo bindings. The goal-conditioned interface is preserved.

Observation Space (4 dim): End-Effector + Puck Position 
Goal Space (4 dim): End-Effector + Puck Position
Action Space (2 dim): End-Effector Position Control
"""

from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict
import mujoco  # Modern mujoco instead of mujoco_py
import copy
import os.path as osp

# Use modern base class instead of multiworld
from modern_envs.core import ModernMujocoEnv, GymGoalEnvWrapper

# Use standalone utilities (no multiworld/gcsl dependencies)
from modern_envs.utils import (
    Serializable,
    MultitaskEnv,
    get_stat_in_paths,
    create_stats_ordered_dict,
)


push_configs = {
    'all': dict(
            hand_low=(-0.20, 0.50),
            hand_high=(0.20, 0.70),
            puck_low=(-0.20, 0.50),
            puck_high=(0.20, 0.70),
            fix_reset=0.075,
            sample_realistic_goals=True,
            invisible_boundary_wall=True,
            reward_type='puck_and_hand',
        ),
    
    'fixed_start': dict(
            hand_low=(-0.20, 0.50),
            hand_high=(0.20, 0.70),
            puck_low=(-0.20, 0.50),
            puck_high=(0.20, 0.70),
            reset_low=(-0.20,  0.50, -0.17, 0.53),
            reset_high=(-0.17, 0.53, -0.10, 0.60),
            fix_reset=False,
            sample_realistic_goals=True,
            invisible_boundary_wall=True,
            reward_type='puck_and_hand',
    ), 
    'fixed_start_fixed_goal': dict(
            hand_low=(-0.20, 0.50),
            hand_high=(0.20, 0.70),
            puck_low=(-0.20, 0.50),
            puck_high=(0.20, 0.70),
            reset_low=(-0.20,  0.50, -0.17, 0.53),
            reset_high=(-0.17, 0.53, -0.10, 0.60),
            fix_reset=False,
            goal_low=(-0.20, 0.50, 0.15, 0.65),
            goal_high=(0.20, 0.70, 0.20, 0.70),
            sample_realistic_goals=True,
            invisible_boundary_wall=True,
            reward_type='puck_and_hand',
    )
}


class SawyerViews:
    """Camera view configurations for rendering."""
    @staticmethod
    def configure_viewer(cam, cam_pos):
        for i in range(3):
            cam.lookat[i] = cam_pos[i]
        cam.distance = cam_pos[3]
        cam.elevation = cam_pos[4]
        cam.azimuth = cam_pos[5]
        cam.trackbodyid = -1
    
    @staticmethod
    def robot_view(cam):
        rotation_angle = 90
        cam_dist = 1
        cam_pos = np.array([0, 0.5, 0.2, cam_dist, -45, rotation_angle])
        SawyerViews.configure_viewer(cam, cam_pos)
    
    @staticmethod
    def third_person_view(cam):
        cam_dist = 0.3
        rotation_angle = 270
        cam_pos = np.array([0, 1.0, 0.5, cam_dist, -45, rotation_angle])
        SawyerViews.configure_viewer(cam, cam_pos)
    
    @staticmethod
    def top_down_view(cam):
        cam_dist = 0.2
        rotation_angle = 0
        cam_pos = np.array([0, 0, 1.5, cam_dist, -90, rotation_angle])
        SawyerViews.configure_viewer(cam, cam_pos)
    
    @staticmethod
    def default_view(cam):
        cam_dist = 0.3
        rotation_angle = 270
        cam_pos = np.array([0, 0.85, 0.30, cam_dist, -55, rotation_angle])
        SawyerViews.configure_viewer(cam, cam_pos)


class SawyerPushAndReachXYEnvModern(ModernMujocoEnv, Serializable, MultitaskEnv):
    """
    Modern MuJoCo implementation of the Sawyer pushing task.
    
    This environment simulates a Sawyer robot arm pushing a puck on a table.
    The goal is to move both the end-effector and the puck to desired positions.
    """
    
    def __init__(
            self,
            frame_skip=20,
            action_scale=2./100,

            hand_low=(-0.2, 0.5),
            hand_high=(0.2, 0.7),

            puck_low=(-0.2, 0.5),
            puck_high=(0.2, 0.7),

            fix_goal=False,
            sample_realistic_goals=False,
            fixed_goal=(-0.05, 0.6, 0.05, 0.6),
            goal_low=None,
            goal_high=None,

            fix_reset=False,
            fixed_reset=(0, 0.55, 0.0, 0.65),
            reset_low=None,
            reset_high=None,

            hand_z_position=0.06,
            puck_z_position=0.02,

            reward_type='state_distance',
            norm_order=2,
            indicator_threshold=0.06,

            num_mocap_calls_for_reset=250,

            square_puck=False,
            heavy_puck=False,
            invisible_boundary_wall=False,

            indicator_threshold_2=0.08,
            indicator_threshold_3=0.12,
    ):
        self.quick_init(locals())

        self.square_puck = square_puck
        self.heavy_puck = heavy_puck
        self.invisible_boundary_wall = invisible_boundary_wall

        if self.invisible_boundary_wall:
            model_name = osp.abspath(osp.join(osp.dirname(__file__), 'assets/push.xml'))
        else:
            raise NotImplementedError()

        # Initialize with modern MuJoCo base
        ModernMujocoEnv.__init__(self, model_name, frame_skip=frame_skip)

        hand_low = np.array(hand_low)
        hand_high = np.array(hand_high)
        mocap_low = hand_low
        mocap_high = hand_high
        self.mocap_low = np.hstack((mocap_low, np.array([0.0])))
        self.mocap_high = np.hstack((mocap_high, np.array([0.5])))
        puck_low = np.array(puck_low)
        puck_high = np.array(puck_high)

        if self.square_puck:
            self.puck_radius = np.sqrt(2) * 0.04
        else:
            self.puck_radius = 0.04

        self.ee_radius = 0.015

        self.obs_space = Box(
            np.hstack((hand_low, puck_low)),
            np.hstack((hand_high, puck_high)),
            dtype=np.float32
        )
        self.hand_space = Box(hand_low, hand_high, dtype=np.float32)
        self.puck_space = Box(puck_low, puck_high, dtype=np.float32)
        
        if goal_low is None:
            goal_low = self.obs_space.low.copy()
        if goal_high is None:
            goal_high = self.obs_space.high.copy()
        if reset_low is None:
            reset_low = self.obs_space.low.copy()
        if reset_high is None:
            reset_high = self.obs_space.high.copy()
            
        goal_low = np.array(goal_low)
        goal_high = np.array(goal_high)
        reset_low = np.array(reset_low)
        reset_high = np.array(reset_high)
        
        self.goal_space = Box(goal_low, goal_high, dtype=np.float32)
        self.reset_space = Box(reset_low, reset_high, dtype=np.float32)
        
        self.observation_space = Dict([
            ('observation', self.obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.obs_space),
            ('state_observation', self.obs_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.obs_space),
            ('proprio_observation', self.hand_space),
            ('proprio_desired_goal', Box(goal_low[:2], goal_high[:2], dtype=np.float32)),
            ('proprio_achieved_goal', self.hand_space),
        ])

        self.num_mocap_calls_for_reset = num_mocap_calls_for_reset

        self.fix_reset = fix_reset
        self.sample_realistic_goals = sample_realistic_goals
        self.fixed_reset = np.array(fixed_reset)
        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self._state_goal = None

        self.reward_type = reward_type
        self.norm_order = norm_order
        self.indicator_threshold = indicator_threshold
        self.indicator_threshold_2 = indicator_threshold_2
        self.indicator_threshold_3 = indicator_threshold_3

        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)
        self._action_scale = action_scale
        self.hand_z_position = hand_z_position
        self.puck_z_position = puck_z_position
        self.reset_counter = 0
        self.reset()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 1.0

    def step(self, action):
        action = np.clip(action, -1, 1)
        mocap_delta_z = self.hand_z_position - self.data.mocap_pos[0, 2]
        new_mocap_action = np.hstack((
            action,
            np.array([mocap_delta_z])
        ))
        self.mocap_set_action(new_mocap_action[:3] * self._action_scale)
        u = np.zeros(7)
        self.do_simulation(u, self.frame_skip)
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        info = self._get_info()
        done = False
        return ob, reward, done, info

    def mocap_set_action(self, action):
        """Set mocap position for end-effector control."""
        pos_delta = action[None]
        new_mocap_pos = self.data.mocap_pos.copy() + pos_delta
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high
        )
        # Modern MuJoCo: direct assignment to mocap_pos
        self.data.mocap_pos[0, :] = new_mocap_pos[0, :]
        self.data.mocap_quat[0, :] = np.array([1, 0, 1, 0])

    def _get_info(self):
        hand_goal = self._state_goal[:2]
        puck_goal = self._state_goal[-2:]
        hand_pos = self.get_endeff_pos()[:2]
        puck_pos = self.get_puck_pos()[:2]

        # Hand distance
        hand_diff = hand_goal - hand_pos
        hand_distance = np.linalg.norm(hand_diff, ord=self.norm_order)
        hand_distance_l2 = np.linalg.norm(hand_diff, 2)

        # Puck distance
        puck_diff = puck_goal - puck_pos
        puck_distance = np.linalg.norm(puck_diff, ord=self.norm_order)
        puck_distance_l2 = np.linalg.norm(puck_diff, 2)

        # Touch distance
        touch_diff = self.get_endeff_pos() - self.get_puck_pos()
        touch_distance = np.linalg.norm(touch_diff, ord=self.norm_order)
        touch_distance_l2 = np.linalg.norm(touch_diff, ord=2)

        # State distance
        state_diff = np.hstack((hand_pos, puck_pos)) - self._state_goal
        state_distance = np.linalg.norm(state_diff, ord=self.norm_order)
        state_distance_l2 = np.linalg.norm(state_diff, ord=2)

        return dict(
            hand_distance=hand_distance, hand_distance_l2=hand_distance_l2,
            puck_distance=puck_distance, puck_distance_l2=puck_distance_l2,
            touch_distance=touch_distance, touch_distance_l2=touch_distance_l2,
            state_distance=state_distance, state_distance_l2=state_distance_l2,
            hand_success=float(hand_distance < self.indicator_threshold),
            puck_success=float(puck_distance < self.indicator_threshold),
            hand_and_puck_success=float(
                hand_distance+puck_distance < self.indicator_threshold
            ),
            touch_success=float(touch_distance < self.indicator_threshold),
            state_success=float(state_distance < self.indicator_threshold),
            hand_success_2=float(hand_distance < self.indicator_threshold_2),
            puck_success_2=float(puck_distance < self.indicator_threshold_2),
            hand_and_puck_success_2=float(
                hand_distance + puck_distance < self.indicator_threshold_2
            ),
            touch_success_2=float(touch_distance < self.indicator_threshold_2),
            state_success_2=float(state_distance < self.indicator_threshold_2),
            hand_success_3=float(hand_distance < self.indicator_threshold_3),
            puck_success_3=float(puck_distance < self.indicator_threshold_3),
            hand_and_puck_success_3=float(
                hand_distance + puck_distance < self.indicator_threshold_3
            ),
            touch_success_3=float(touch_distance < self.indicator_threshold_3),
            state_success_3=float(state_distance < self.indicator_threshold_3),
        )

    def _get_obs(self):
        e = self.get_endeff_pos()[:2]
        b = self.get_puck_pos()[:2]
        flat_obs = np.concatenate((e, b))

        return dict(
            observation=flat_obs,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs,
            proprio_observation=flat_obs[:2],
            proprio_desired_goal=self._state_goal[:2],
            proprio_achieved_goal=flat_obs[:2],
        )

    def compute_rewards(self, actions, obs, prev_obs=None, reward_type=None):
        if reward_type is None:
            reward_type = self.reward_type

        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        hand_pos = achieved_goals[:, :2]
        puck_pos = achieved_goals[:, -2:]
        hand_goals = desired_goals[:, :2]
        puck_goals = desired_goals[:, -2:]

        hand_distances = np.linalg.norm(hand_goals - hand_pos, ord=self.norm_order, axis=1)
        puck_distances = np.linalg.norm(puck_goals - puck_pos, ord=self.norm_order, axis=1)
        touch_distances = np.linalg.norm(hand_pos - puck_pos, ord=self.norm_order, axis=1)

        if reward_type == 'hand_distance':
            r = -hand_distances
        elif reward_type == 'hand_success':
            r = -(hand_distances > self.indicator_threshold).astype(float)
        elif reward_type == 'puck_distance':
            r = -puck_distances
        elif reward_type == 'puck_success':
            r = -(puck_distances > self.indicator_threshold).astype(float)
        elif reward_type == 'puck_and_hand':
            r = - hand_distances - 5 * puck_distances
        elif reward_type == 'vectorized_puck_distance':
            r = -np.abs(puck_goals - puck_pos)
        elif reward_type == 'state_distance':
            r = -np.linalg.norm(
                achieved_goals - desired_goals,
                ord=self.norm_order,
                axis=1
            )
        elif reward_type == 'vectorized_state_distance':
            r = -np.abs(achieved_goals - desired_goals)
        elif reward_type == 'touch_distance':
            r = -touch_distances
        elif reward_type == 'touch_success':
            r = -(touch_distances > self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def compute_reward(self, action, obs):
        """Compute reward for single step (compatibility method)."""
        # Expand dims to batch format expected by compute_rewards
        obs_batch = {k: np.array([v]) for k, v in obs.items()}
        return self.compute_rewards(np.array([action]), obs_batch)[0]

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance', 'hand_distance_l2',
            'puck_distance', 'puck_distance_l2',
            'state_distance', 'state_distance_l2',
            'touch_distance', 'touch_distance_l2',
            'hand_success', 'hand_success_2', 'hand_success_3',
            'puck_success', 'puck_success_2', 'puck_success_3',
            'hand_and_puck_success', 'hand_and_puck_success_2', 'hand_and_puck_success_3',
            'state_success', 'state_success_2', 'state_success_3',
            'touch_success', 'touch_success_2', 'touch_success_3',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        return statistics

    def get_puck_pos(self):
        """Get puck position using modern MuJoCo API."""
        # In modern mujoco: body_xpos is accessed via data
        return self.data.xpos[self.puck_id].copy()

    def get_endeff_pos(self):
        """Get end-effector position using modern MuJoCo API."""
        return self.data.xpos[self.endeff_id].copy()

    @property
    def endeff_id(self):
        """Get end-effector body ID using modern MuJoCo API."""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'leftclaw')

    @property
    def puck_id(self):
        """Get puck body ID using modern MuJoCo API."""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'puck')

    def reset(self):
        self.subgoals = None
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def reset_model(self):
        self._reset_hand()
        self._reset_puck()

        goal = self._sample_realistic_goal()
        self.set_goal(goal)
        self.reset_counter += 1
        self.reset_mocap_welds()
        return self._get_obs()

    def _reset_hand(self):
        velocities = self.data.qvel.copy()
        angles = np.array(self.init_angles)
        self.set_state(angles.flatten(), velocities.flatten())

        if self.fix_reset is True:
            new_mocap_pos_xy = self.fixed_reset[:2].copy()
        elif self.fix_reset is False:
            new_mocap_pos_xy = np.random.uniform(self.reset_space.low[:2], self.reset_space.high[:2])
        else:
            new_mocap_pos_xy = np.random.uniform(self.reset_space.low[:2], self.reset_space.high[:2])
            
        new_mocap_pos = np.hstack((new_mocap_pos_xy, np.array([self.hand_z_position])))
        
        for i in range(self.num_mocap_calls_for_reset):
            # Modern MuJoCo: direct assignment
            self.data.mocap_pos[0, :] = new_mocap_pos
            self.data.mocap_quat[0, :] = np.array([1, 0, 1, 0])
            self.do_simulation(None, 20)
            e = self.get_endeff_pos().copy()
            if np.linalg.norm(new_mocap_pos - e) < .002 or i > 50:
                break

    def _reset_puck(self):
        puck_xy = self.sample_puck_xy()
        while self.end_effector_puck_collision(self.get_endeff_pos()[:2], puck_xy):
            puck_xy = self.sample_puck_xy()
        self._set_puck_xy(puck_xy)

    def sample_puck_xy(self):
        if self.fix_reset is True:
            return self.fixed_reset[-2:].copy()
        elif self.fix_reset is False:
            return np.random.uniform(self.reset_space.low[-2:], self.reset_space.high[-2:])
        else:
            max_radius = self.fix_reset
            radius = np.random.uniform(self.ee_radius + self.puck_radius, max_radius)
            angle = np.pi * np.random.uniform(0, 2)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            puck_xy = np.array([x, y]) + self.get_endeff_pos()[:2]
            puck_xy = np.clip(puck_xy, self.puck_space.low, self.puck_space.high)
            return puck_xy

    def end_effector_puck_collision(self, ee, puck):
        dist = np.linalg.norm(ee - puck)
        return dist <= (self.puck_radius + self.ee_radius)

    def realistic_state_np(self, state):
        return not self.end_effector_puck_collision(state[:2], state[2:])

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal']
        hand_goal = self._state_goal[:2]
        puck_goal = self._state_goal[-2:]
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[14:17] = np.hstack((hand_goal.copy(), np.array([self.hand_z_position])))
        qvel[14:17] = [0, 0, 0]
        qpos[21:24] = np.hstack((puck_goal.copy(), np.array([self.puck_z_position])))
        qvel[21:24] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        self._set_hand_xy(state_goal[:2])
        self._set_puck_xy(state_goal[-2:])

    def _set_hand_xy(self, xy):
        for _ in range(10):
            # Modern MuJoCo: direct assignment
            self.data.mocap_pos[0, :] = np.array([xy[0], xy[1], self.hand_z_position])
            self.data.mocap_quat[0, :] = np.array([1, 0, 1, 0])
            u = np.zeros(7)
            self.do_simulation(u, self.frame_skip)

    def _set_puck_xy(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[7:10] = np.hstack((pos.copy(), np.array([self.puck_z_position])))
        qvel[7:10] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def _sample_realistic_goal(self):
        if self.fix_goal:
            goal = self.fixed_goal.copy()
        else:
            dist = -1
            goal = None
            while (dist <= self.puck_radius + self.ee_radius):
                goal = np.random.uniform(self.goal_space.low, self.goal_space.high)
                hand_pos = goal[:2]
                puck_pos = goal[-2:]
                dist = np.linalg.norm(hand_pos - puck_pos)
        return {
            'desired_goal': goal,
            'state_desired_goal': goal,
        }

    def sample_goals(self, batch_size):
        if self.fix_goal:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        elif self.sample_realistic_goals:
            goals = np.array([
                self._sample_realistic_goal()['state_desired_goal']
                for _ in range(batch_size)
            ])
        else:
            goals = np.random.uniform(
                self.goal_space.low,
                self.goal_space.high,
                size=(batch_size, self.goal_space.low.size),
            )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def get_env_state(self):
        """Get full environment state for saving/loading."""
        qpos, qvel = self.get_state()
        mocap_state = self.data.mocap_pos.copy(), self.data.mocap_quat.copy()
        base_state = (qpos, qvel), mocap_state
        base_state = copy.deepcopy(base_state)
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        """Restore environment state."""
        base_state, goal = state
        (qpos, qvel), mocap_state = base_state
        self.set_state(qpos, qvel)
        mocap_pos, mocap_quat = mocap_state
        # Modern MuJoCo: direct assignment
        self.data.mocap_pos[:] = mocap_pos
        self.data.mocap_quat[:] = mocap_quat
        self.set_goal({'state_desired_goal': goal})

    def reset_mocap_welds(self):
        """
        Reset mocap welds for actuation.
        
        CHANGE: Use modern MuJoCo constant and handle eq_data size difference
        """
        if self.model.nmocap > 0 and self.model.eq_data is not None:
            for i in range(self.model.eq_data.shape[0]):
                # Modern MuJoCo: mujoco.mjtEq.mjEQ_WELD instead of mujoco_py.const.EQ_WELD
                if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    # Modern MuJoCo eq_data has 11 values, not 7
                    # We only need to set the first 7 (pos + quat)
                    self.model.eq_data[i, :7] = np.array([0., 0., 0., 1., 0., 0., 0.])
        mujoco.mj_forward(self.model, self.data)

    @property
    def init_angles(self):
        return [1.78026069e+00, - 6.84415781e-01, - 1.54549231e-01,
                2.30672090e+00, 1.93111471e+00, 1.27854012e-01,
                1.49353907e+00, 1.80196716e-03, 7.40415706e-01,
                2.09895360e-02, 9.99999990e-01, 3.05766105e-05,
                - 3.78462492e-06, 1.38684523e-04, - 3.62518873e-02,
                6.13435141e-01, 2.09686080e-02, 7.07106781e-01,
                1.48979724e-14, 7.07106781e-01, - 1.48999170e-14,
                0, 0.6, 0.02,
                1, 0, 1, 0,
                ]


class SawyerPushGoalEnv(GymGoalEnvWrapper):
    """
    Goal-conditioned wrapper for SawyerPush environment.
    
    This wrapper converts the multiworld-style environment into the
    GoalEnv format expected by GCSL/HDM algorithms.
    """
    
    def __init__(self, fixed_start=True, fixed_goal=False, images=False, image_kwargs=None):
        config_key = 'all'
        if fixed_start:
            if fixed_goal:
                config_key = 'fixed_start_fixed_goal'
            else:
                config_key = 'fixed_start'
        
        # Use modern environment
        env = SawyerPushAndReachXYEnvModern(**push_configs[config_key])
        
        if images:
            # TODO: Port ImageEnv if needed for vision-based experiments
            raise NotImplementedError(
                "Image observations not yet implemented in modern version. "
                "Set images=False to use state-based observations."
            )

        super(SawyerPushGoalEnv, self).__init__(
            env, 
            observation_key='observation', 
            goal_key='achieved_goal', 
            state_goal_key='state_achieved_goal'
        )
    
    def endeff_distance(self, states, goal_states):
        achieved_goals = self._extract_sgoal(states)
        desired_goals = self._extract_sgoal(goal_states)
        diff = achieved_goals - desired_goals
        return np.linalg.norm(diff[..., 0:2], axis=-1)
    
    def goal_distance(self, states, goal_states):
        return self.puck_distance(states, goal_states)
    
    def puck_distance(self, states, goal_states):
        achieved_goals = self._extract_sgoal(states)
        desired_goals = self._extract_sgoal(goal_states)
        diff = achieved_goals - desired_goals
        return np.linalg.norm(diff[..., 2:4], axis=-1)

    def get_diagnostics(self, trajectories, desired_goal_states):
        """Log diagnostics for evaluation."""
        endeff_distances = np.array([
            self.endeff_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1],1))) 
            for i in range(trajectories.shape[0])
        ])
        puck_distances = np.array([
            self.puck_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1],1))) 
            for i in range(trajectories.shape[0])
        ])

        endeff_movement = self.endeff_distance(trajectories[:,0], trajectories[:, -1])
        puck_movement = self.puck_distance(trajectories[:,0], trajectories[:, -1])
        
        statistics = OrderedDict()
        for stat_name, stat in [
            ('final puck distance', puck_distances[:,-1]),
            ('final endeff distance', endeff_distances[:,-1]),
            ('puck movement', puck_movement),
            ('endeff movement', endeff_movement),
        ]:
            statistics.update(create_stats_ordered_dict(
                    stat_name,
                    stat,
                    always_show_all_stats=True,
                ))
            
        return statistics

