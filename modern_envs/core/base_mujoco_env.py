"""
Modern MuJoCo base environment using mujoco (not mujoco_py).

This is a replacement for multiworld's MujocoEnv that uses the modern
mujoco Python bindings instead of the deprecated mujoco_py.

Key API Changes from mujoco_py:
================================
OLD (mujoco_py)                    →  NEW (mujoco)
-------------------------------------------------------------------
mujoco_py.load_model_from_path()   →  mujoco.MjModel.from_xml_path()
mujoco_py.MjSim(model)             →  mujoco.MjData(model)
sim.data.qpos                      →  data.qpos
sim.data.qvel                      →  data.qvel
sim.data.ctrl                      →  data.ctrl
sim.step()                         →  mujoco.mj_step(model, data)
sim.forward()                      →  mujoco.mj_forward(model, data)
sim.reset()                        →  mujoco.mj_resetData(model, data)
sim.get_state() / set_state()      →  Direct manipulation of data.qpos/qvel
mujoco_py.MjViewer(sim)            →  Custom viewer or passive_viewer
sim.render()                       →  mujoco.Renderer with manual setup
"""

import os
import numpy as np
from gym import spaces
from gym.utils import seeding
from os import path
import gym

try:
    import mujoco
except ImportError as e:
    raise ImportError(
        f"{e}. (HINT: Install modern mujoco with: pip install mujoco)"
    )


class ModernMujocoEnv(gym.Env):
    """
    Base class for MuJoCo environments using modern mujoco bindings.
    
    This is designed as a drop-in replacement for multiworld's MujocoEnv,
    maintaining the same interface so existing environments can migrate easily.
    
    Key differences from multiworld.MujocoEnv:
    - Uses mujoco instead of mujoco_py
    - Renderer is lazily initialized and separate from physics
    - State management is simplified (direct qpos/qvel access)
    """
    
    def __init__(self, model_path, frame_skip, device_id=-1, 
                 automatically_set_spaces=False):
        """
        Initialize the MuJoCo environment.
        
        Args:
            model_path: Path to the XML model file (absolute or relative to assets/)
            frame_skip: Number of physics steps per environment step
            device_id: GPU device ID (not used in modern mujoco, kept for compatibility)
            automatically_set_spaces: If True, infer action/obs spaces from model
        """
        # Resolve model path
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError(f"File {fullpath} does not exist")
        
        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.data = mujoco.MjData(self.model)
        
        self.frame_skip = frame_skip
        self.device_id = device_id  # Kept for compatibility, not used
        
        # Rendering components (lazily initialized)
        self.viewer = None
        self._renderer = None
        self._render_context = None
        
        # Store initial state
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        
        # Metadata for gym
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        
        # Optionally auto-set observation and action spaces
        if automatically_set_spaces:
            observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
            assert not done
            self.obs_dim = observation.size

            bounds = self.model.actuator_ctrlrange.copy()
            low = bounds[:, 0]
            high = bounds[:, 1]
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        self.seed()

    def seed(self, seed=None):
        """Set random seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # ========================================================================
    # Methods to override in subclasses
    # ========================================================================

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Must be implemented in each subclass.
        
        Returns:
            observation: Initial observation after reset
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        Configure camera position and settings.
        Called when viewer is initialized and after every reset.
        Override in subclasses if needed.
        """
        pass

    # ========================================================================
    # Core environment methods
    # ========================================================================

    def reset(self):
        """Reset the environment."""
        mujoco.mj_resetData(self.model, self.data)
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        """
        Set the joint positions and velocities.
        
        Args:
            qpos: Joint positions (shape: [nq])
            qvel: Joint velocities (shape: [nv])
        """
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def get_state(self):
        """
        Get current simulation state.
        
        Returns:
            tuple: (qpos, qvel) arrays
        """
        return self.data.qpos.copy(), self.data.qvel.copy()

    @property
    def dt(self):
        """Timestep duration."""
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames=None):
        """
        Step the simulation forward.
        
        Args:
            ctrl: Control inputs (can be None for passive simulation)
            n_frames: Number of frames to simulate (default: self.frame_skip)
        """
        if n_frames is None:
            n_frames = self.frame_skip
        
        if self.data.ctrl is not None and ctrl is not None:
            self.data.ctrl[:] = ctrl
        
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)

    # ========================================================================
    # Rendering methods
    # ========================================================================

    def render(self, mode='human', width=500, height=500, camera_id=None):
        """
        Render the environment.
        
        Args:
            mode: 'human' for window display, 'rgb_array' for numpy array
            width: Image width for rgb_array mode
            height: Image height for rgb_array mode
            camera_id: Camera ID to use (None for free camera)
            
        Returns:
            RGB image array if mode='rgb_array', else None
        """
        if mode == 'rgb_array':
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            
            self._renderer.update_scene(self.data, camera=camera_id)
            pixels = self._renderer.render()
            # Modern mujoco returns correct orientation
            return pixels
            
        elif mode == 'human':
            # For human mode, we'd typically use mujoco.viewer.launch_passive
            # But for compatibility with the old API, we'll use a simple approach
            if self.viewer is None:
                # Use passive viewer - this requires the environment to keep running
                print("Warning: 'human' render mode not fully implemented. Use 'rgb_array' instead.")
            return None

    def close(self):
        """Clean up rendering resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self.viewer is not None:
            self.viewer = None

    # ========================================================================
    # Utility methods
    # ========================================================================

    def get_body_com(self, body_name):
        """
        Get center of mass position for a body.
        
        Args:
            body_name: Name of the body
            
        Returns:
            3D position vector
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[body_id].copy()

    def state_vector(self):
        """Get full state as concatenated qpos and qvel."""
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat
        ])

    def get_image(self, width=84, height=84, camera_name=None):
        """
        Render an image from a camera view.
        
        Args:
            width: Image width
            height: Image height
            camera_name: Name of camera in model (None for free camera)
            
        Returns:
            RGB image array
        """
        if camera_name is not None:
            camera_id = mujoco.mj_name2id(
                self.model, 
                mujoco.mjtObj.mjOBJ_CAMERA, 
                camera_name
            )
        else:
            camera_id = None
        
        return self.render(mode='rgb_array', width=width, height=height, camera_id=camera_id)

    def initialize_camera(self, init_fctn):
        """
        Initialize camera for rendering (compatibility method).
        
        Args:
            init_fctn: Function that takes a camera object and configures it
        
        Note: Modern mujoco handles cameras differently. This method provides
        basic compatibility but may not support all mujoco_py features.
        """
        # Create renderer if needed
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model)
        
        # Modern mujoco cameras are configured differently
        # We'll store the init function and apply it during rendering
        self._camera_init_fctn = init_fctn
        
        # Try to apply camera settings
        # Note: This is a simplified version and may need adjustment
        # based on what init_fctn actually does
        if hasattr(init_fctn, '__call__'):
            # Create a camera-like object for compatibility
            class CameraProxy:
                def __init__(self, renderer):
                    self.lookat = np.zeros(3)
                    self.distance = 1.0
                    self.elevation = -45
                    self.azimuth = 0
                    self.trackbodyid = -1
            
            cam_proxy = CameraProxy(self._renderer)
            try:
                init_fctn(cam_proxy)
            except Exception as e:
                print(f"Warning: Camera initialization had issues: {e}")

    # ========================================================================
    # Additional compatibility methods
    # ========================================================================

    @property
    def sim(self):
        """
        Compatibility property to access 'sim' like in mujoco_py.
        In modern mujoco, we work directly with model and data.
        """
        # Return a proxy object that provides sim-like interface
        class SimProxy:
            def __init__(self, env):
                self.env = env
                self.model = env.model
                self.data = env.data
            
            def step(self):
                mujoco.mj_step(self.model, self.data)
            
            def forward(self):
                mujoco.mj_forward(self.model, self.data)
            
            def reset(self):
                mujoco.mj_resetData(self.model, self.data)
            
            def get_state(self):
                return self.env.get_state()
            
            def set_state(self, state):
                qpos, qvel = state
                self.env.set_state(qpos, qvel)
        
        return SimProxy(self)

