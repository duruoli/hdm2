"""
Modern SawyerDoor environment using mujoco (not mujoco_py).

This is a port of the original GCSL SawyerDoor environment to use
modern MuJoCo bindings. The goal-conditioned interface is preserved.

Observation Space (4 dim): End-Effector Position + Door Angle
Goal Space (1 dim): Door Angle
Action Space (3 dim): End-Effector Position Control
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


door_configs = {
    'all': dict(
            goal_low=(0,),
            goal_high=(.83,),
            hand_low=(-0.1, 0.45, 0.1),
            hand_high=(0.05, 0.65, .25),
            max_angle=.83,
            reward_type='angle_diff_and_hand_distance',
            reset_free=False,
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


class SawyerDoorHookEnvModern(ModernMujocoEnv, Serializable, MultitaskEnv):
    """
    Modern MuJoCo implementation of the Sawyer door opening task.
    
    This environment simulates a Sawyer robot arm opening a door with a hook handle.
    The goal is to rotate the door to a desired angle.
    """
    
    def __init__(
            self,
            frame_skip=50,
            action_scale=2./100,
            
            goal_low=(0,),
            goal_high=(1.0472,),
            
            hand_low=(-0.1, 0.45, 0.15),
            hand_high=(0., 0.65, .225),
            
            mocap_low=None,
            mocap_high=None,
            
            fix_goal=False,
            fixed_goal=(-.25,),
            
            reset_free=False,
            fixed_hand_z=0.12,
            
            reward_type='angle_difference',
            indicator_threshold=(.02, .03),
            
            min_angle=0,
            max_angle=1.0472,
            
            action_reward_scale=0,
            target_pos_scale=1,
            target_angle_scale=1,
    ):
        self.quick_init(locals())
        
        # Get XML path
        model_name = osp.abspath(osp.join(osp.dirname(__file__), '../assets/sawyer_door_pull_hook.xml'))
        
        # Initialize with modern MuJoCo base
        ModernMujocoEnv.__init__(self, model_name, frame_skip=frame_skip)
        MultitaskEnv.__init__(self)
        
        # Action scale
        self.action_scale = action_scale
        
        # Hand/mocap bounds
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        if mocap_low is None:
            mocap_low = hand_low
        if mocap_high is None:
            mocap_high = hand_high
        self.mocap_low = np.array(mocap_low)
        self.mocap_high = np.array(mocap_high)
        
        # Goal configuration
        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float32)
        self._state_goal = None
        
        # State space: hand position (3D) + door angle (1D)
        self.state_space = Box(
            np.concatenate((self.hand_low, [min_angle])),
            np.concatenate((self.hand_high, [max_angle])),
            dtype=np.float32,
        )
        
        # Observation space
        self.observation_space = Dict([
            ('observation', self.state_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.state_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
        ])
        
        # Action space: XYZ control
        self.action_space = Box(np.array([-1, -1, -1]), np.array([1, 1, 1]), dtype=np.float32)
        
        # Reward configuration
        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold
        self.action_reward_scale = action_reward_scale
        self.target_pos_scale = target_pos_scale
        self.target_angle_scale = target_angle_scale
        
        # Door angle tracking
        self.door_angle_idx = None
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name == 'doorjoint':
                # Get qpos address for this joint
                self.door_angle_idx = self.model.jnt_qposadr[i]
                break
        
        if self.door_angle_idx is None:
            raise ValueError("Could not find 'doorjoint' in model")
        
        # Reset configuration
        self.fixed_hand_z = fixed_hand_z
        self.reset_free = reset_free
        
        # Ensure env does not start in weird positions
        temp_reset_free = self.reset_free
        self.reset_free = True
        self.reset()
        self.reset_free = temp_reset_free

    def viewer_setup(self):
        """Configure camera view."""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.cam.trackbodyid = -1
            self.viewer.cam.lookat[0] = -.2
            self.viewer.cam.lookat[1] = .55
            self.viewer.cam.lookat[2] = 0.6
            self.viewer.cam.distance = 0.25
            self.viewer.cam.elevation = -60
            self.viewer.cam.azimuth = 360

    def step(self, action):
        """Execute action and return observation."""
        self.set_xyz_action(action)
        u = np.zeros(7)
        self.do_simulation(u, self.frame_skip)
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        info = self._get_info()
        done = False
        return ob, reward, done, info

    def set_xyz_action(self, action):
        """Set end-effector action using mocap."""
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos.copy() + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        # Modern MuJoCo: direct assignment
        self.data.mocap_pos[0, :] = new_mocap_pos[0, :]
        self.data.mocap_quat[0, :] = np.array([1, 0, 1, 0])

    def _get_obs(self):
        """Get current observation."""
        pos = self.get_endeff_pos()
        angle = self.get_door_angle()
        flat_obs = np.concatenate((pos, angle))
        return dict(
            observation=flat_obs,
            desired_goal=self._state_goal,
            achieved_goal=angle,
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=angle,
        )

    def _get_info(self):
        """Get diagnostic information."""
        angle_diff = np.abs(self.get_door_angle() - self._state_goal[-1])[0]
        info = dict(
            angle_difference=angle_diff,
            angle_success=float(angle_diff < self.indicator_threshold[0]),
        )
        return info

    def get_door_angle(self):
        """Get current door angle."""
        return np.array([self.data.qpos[self.door_angle_idx]])

    def get_endeff_pos(self):
        """Get end-effector position using modern MuJoCo API."""
        return self.data.xpos[self.endeff_id].copy()

    @property
    def endeff_id(self):
        """Get end-effector body ID using modern MuJoCo API."""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'leftclaw')

    def compute_rewards(self, actions, obs):
        """Compute rewards for batch of observations."""
        # Placeholder - can be extended for batch evaluation
        r = np.array([0])
        return r

    def compute_reward(self, action, obs):
        """Compute reward for single step (compatibility method)."""
        return 0.0

    def reset_model(self):
        """Reset the robot to initial configuration."""
        if not self.reset_free:
            self._reset_hand()
            self._set_door_pos(0)
        goal = self.sample_goal()
        self.set_goal(goal)
        self.reset_mocap_welds()
        return self._get_obs()

    def reset(self):
        """Reset environment."""
        self.subgoals = None
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def _reset_hand(self):
        """Reset hand to initial position."""
        velocities = self.data.qvel.copy()
        angles = self.data.qpos.copy()
        # Do this to make sure the robot isn't in some weird configuration
        angles[:7] = self.init_arm_angles
        self.set_state(angles.flatten(), velocities.flatten())
        self._set_hand_pos(np.array([-.05, .635, .225]))

    def _set_hand_pos(self, pos):
        """Set hand to specific position."""
        for _ in range(10):
            # Modern MuJoCo: direct assignment
            self.data.mocap_pos[0, :] = pos
            self.data.mocap_quat[0, :] = np.array([1, 0, 1, 0])
            u = np.zeros(7)
            self.do_simulation(u, self.frame_skip)

    @property
    def init_arm_angles(self):
        """Initial arm joint angles."""
        return [1.7244448, -0.92036369, 0.10234232, 2.11178144, 
                2.97668632, -0.38664629, 0.54065733]

    def _set_door_pos(self, pos):
        """Set door to specific angle."""
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[self.door_angle_idx] = pos
        qvel[self.door_angle_idx] = 0
        self.set_state(qpos.flatten(), qvel.flatten())

    # Multitask Functions
    
    @property
    def goal_dim(self):
        """Goal dimensionality."""
        return 1

    def set_goal(self, goal):
        """Set the current goal."""
        self._state_goal = goal['state_desired_goal']

    def sample_goals(self, batch_size):
        """Sample a batch of goals."""
        if self.fix_goal:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
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

    def sample_goal(self):
        """Sample a single goal."""
        return self.sample_goals(1)

    def set_to_goal_angle(self, angle):
        """Set door to goal angle."""
        self._state_goal = angle.copy()
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[self.door_angle_idx] = angle.copy()
        qvel[self.door_angle_idx] = 0
        self.set_state(qpos, qvel)

    def set_to_goal_pos(self, xyz):
        """Set hand to goal position."""
        for _ in range(10):
            self.data.mocap_pos[0, :] = np.array(xyz)
            self.data.mocap_quat[0, :] = np.array([1, 0, 1, 0])
            u = np.zeros(7)
            self.do_simulation(u, self.frame_skip)

    def get_goal(self):
        """Get current goal."""
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def set_to_goal(self, goal):
        """Set environment to goal state."""
        raise NotImplementedError("Hard to do because what if the hand is in "
                                  "the door? Use presampled goals.")

    def get_diagnostics(self, paths, prefix=''):
        """Get diagnostic statistics from paths."""
        statistics = OrderedDict()
        for stat_name in [
            'angle_difference',
            'angle_success'
        ]:
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
        self._state_goal = goal

    def reset_mocap_welds(self):
        """
        Reset mocap welds for actuation.
        
        Uses modern MuJoCo constant and handles eq_data size difference.
        """
        if self.model.nmocap > 0 and self.model.eq_data is not None:
            for i in range(self.model.eq_data.shape[0]):
                # Modern MuJoCo: mujoco.mjtEq.mjEQ_WELD instead of mujoco_py.const.EQ_WELD
                if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    # Modern MuJoCo eq_data has 11 values, not 7
                    # We only need to set the first 7 (pos + quat)
                    self.model.eq_data[i, :7] = np.array([0., 0., 0., 1., 0., 0., 0.])
        mujoco.mj_forward(self.model, self.data)


class SawyerDoorGoalEnv(GymGoalEnvWrapper):
    """
    Goal-conditioned wrapper for SawyerDoor environment.
    
    This wrapper converts the multiworld-style environment into the
    GoalEnv format expected by GCSL/HDM algorithms.
    """
    
    def __init__(self, fixed_start=True, fixed_goal=False, images=False, image_kwargs=None):
        config_key = 'all'
        
        # Use modern environment
        env = SawyerDoorHookEnvModern(**door_configs[config_key])
        
        if images:
            # TODO: Port ImageEnv if needed for vision-based experiments
            raise NotImplementedError(
                "Image observations not yet implemented in modern version. "
                "Set images=False to use state-based observations."
            )

        super(SawyerDoorGoalEnv, self).__init__(
            env, 
            observation_key='observation', 
            goal_key='achieved_goal', 
            state_goal_key='state_achieved_goal'
        )
    
    def extract_goal(self, states):
        """Extract goal from states."""
        original_goal = super().extract_goal(states)
        return original_goal
    
    def goal_distance(self, states, goal_states):
        """Compute distance to goal."""
        return self.door_distance(states, goal_states)

    def door_distance(self, states, goal_states):
        """Compute door angle distance."""
        achieved_goals = self._extract_sgoal(states)
        desired_goals = self._extract_sgoal(goal_states)
        diff = achieved_goals - desired_goals
        return np.linalg.norm(diff[..., -1:], axis=-1)

    def get_diagnostics(self, trajectories, desired_goal_states):
        """
        Logs diagnostics.

        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]
        """
        door_distances = np.array([
            self.door_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1],1))) 
            for i in range(trajectories.shape[0])
        ])

        statistics = OrderedDict()
        for stat_name, stat in [
            ('final door distance', door_distances[:,-1]),
            ('min door distance', np.min(door_distances, axis=-1)),
        ]:
            statistics.update(create_stats_ordered_dict(
                    stat_name,
                    stat,
                    always_show_all_stats=True,
                ))
            
        return statistics









