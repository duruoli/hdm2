# Modern MuJoCo Environments for HDM/GCSL

This module provides goal-conditioned robotic environments using **modern MuJoCo** (≥2.2.0) instead of the deprecated `mujoco_py`.

## Why This Module?

The original GCSL/HDM codebase uses:
- `mujoco_py` (deprecated, requires compilation, Python 3.7 only)
- `multiworld` (wrapper library that depends on `mujoco_py`)

This module **replaces only the MuJoCo layer** while keeping all goal-conditioned wrappers intact.

## Architecture

```
┌─────────────────────────────────────┐
│  HDM Algorithm                      │  ← Unchanged
└──────────────┬──────────────────────┘
               │
    ┌──────────▼──────────┐
    │ DiscretizedActionEnv│             ← Unchanged (from GCSL)
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ GymGoalEnvWrapper   │             ← Unchanged (from GCSL)
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ GoalEnv base class  │             ← Unchanged (from GCSL)
    └──────────┬──────────┘
               │
┌──────────────▼───────────────────┐
│ Modern MuJoCo Environments       │   ← NEW (this module)
│  - ModernMujocoEnv (base)        │
│  - SawyerPushModern              │
│  - SawyerDoorModern              │
│  - ClawModern                    │
└──────────────────────────────────┘
```

## Key API Changes: mujoco_py → mujoco

| Old API (mujoco_py) | New API (mujoco) |
|---------------------|------------------|
| `mujoco_py.load_model_from_path(path)` | `mujoco.MjModel.from_xml_path(path)` |
| `mujoco_py.MjSim(model)` | `mujoco.MjData(model)` |
| `sim.step()` | `mujoco.mj_step(model, data)` |
| `sim.forward()` | `mujoco.mj_forward(model, data)` |
| `sim.reset()` | `mujoco.mj_resetData(model, data)` |
| `sim.data.qpos` | `data.qpos` |
| `sim.get_state()` / `set_state()` | Direct `data.qpos`/`qvel` manipulation |
| `mujoco_py.MjViewer(sim)` | `mujoco.Renderer(model)` |
| `sim.render(width, height)` | `renderer.update_scene(data); renderer.render()` |

## Files

- `base_mujoco_env.py` - Base class for all modern MuJoCo environments
- `sawyer_push_modern.py` - Robotic pushing task (planned)
- `sawyer_door_modern.py` - Door opening task (planned)
- `claw_modern.py` - DClaw manipulation (planned)

## Usage

```python
from modern_envs import create_env

# Create environment (will use modern MuJoCo)
env = create_env('pusher')

# Same interface as before
state = env.reset()
next_state, reward, done, info = env.step(action)
```

## Installation Requirements

```bash
# Modern MuJoCo (no compilation needed!)
pip install mujoco>=2.3.0

# Existing dependencies
pip install gym numpy
```

## Status

- ✅ **ModernMujocoEnv** - Base class (complete)
- ⏳ **SawyerPush** - In progress
- ⏳ **SawyerDoor** - Planned
- ⏳ **Claw** - Planned

## Benefits

1. **No compilation** - Modern MuJoCo is pure Python
2. **macOS compatible** - No more M1/M2 compilation issues
3. **Python 3.8+** - Works with modern Python versions
4. **Maintained** - Active development by DeepMind
5. **Faster** - Better performance than mujoco_py

