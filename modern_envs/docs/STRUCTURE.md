# Modern Envs Folder Structure

## Overview

Clean, organized structure with separated concerns:

```
modern_envs/
├── __init__.py               # Main entry: create_env(), get_env_params()
│
├── core/                     # Base classes & interfaces
│   ├── __init__.py
│   ├── base_mujoco_env.py   # Modern MuJoCo base (replaces mujoco_py)
│   ├── goal_env.py          # Goal-conditioned interface
│   └── goal_env_wrapper.py  # Dict→flat state wrapper
│
├── wrappers/                 # Environment wrappers
│   ├── __init__.py
│   └── discretized_action_env.py  # Continuous→discrete actions
│
├── envs/                     # Specific environments
│   ├── __init__.py
│   ├── sawyer_push.py       # ✅ SawyerPush (MuJoCo-based)
│   ├── lunar_lander.py      # ✅ Lunar Lander (Box2D-based)
│   └── lunar_lander_base.py # Lunar Lander physics
│
├── utils/                    # Shared utilities
│   ├── __init__.py
│   └── common.py            # Serializable, MultitaskEnv, logging
│
├── assets/                   # MuJoCo models & meshes
│   ├── push.xml
│   └── meshes/
│       └── sawyer/          # Robot mesh files (.stl)
│
├── tests/                    # All test files
│   ├── test_base.py         # Test base MuJoCo class
│   ├── test_sawyer_push.py  # Test SawyerPush
│   ├── test_lunar.py        # Test Lunar Lander
│   └── test_hdm_integration.py  # Test HDM workflow
│
└── docs/                     # Documentation
    ├── README.md            # Main documentation
    ├── MIGRATION_GUIDE.md   # Migration from mujoco_py
    └── STRUCTURE.md         # This file
```

---

## Available Environments

### ✅ Working
- **`pusher`** (SawyerPush) - Robot pushing task, modern MuJoCo
- **`lunar`** (Lunar Lander) - Rocket landing task, Box2D physics

### ⏳ TODO  
- **`door`** - Door opening task (needs porting)
- **`claw`** - DClaw manipulation (needs porting)
- **`pointmass_rooms`** - Navigation in rooms
- **`pointmass_empty`** - Empty space navigation

---

## Usage

### Basic Usage

```python
from modern_envs import create_env, get_env_params

# Create environment
env = create_env('pusher')  # or 'lunar'
env_params = get_env_params('pusher')

# Use with HDM (automatic discretization)
from modern_envs.wrappers import DiscretizedActionEnv
env = DiscretizedActionEnv(env, granularity=3)
```

### With HDM Training

```bash
# HDM automatically uses modern_envs
python -m hdm --env_name pusher ...
python -m hdm --env_name lunar ...
```

---

## Key Features

### 1. No Dependencies on Old Packages
- ❌ No `mujoco_py` (deprecated, hard to compile)
- ❌ No `gcsl` dependencies for MuJoCo envs
- ❌ No `multiworld` dependencies
- ✅ Pure modern `mujoco` (≥2.3.0)

### 2. Clean Separation
- **Core**: Reusable base classes
- **Wrappers**: Environment transformations
- **Envs**: Specific task implementations
- **Utils**: Shared helper functions
- **Tests**: All tests in one place
- **Docs**: All documentation together

### 3. Organized Assets
- MuJoCo XML models in `assets/`
- Mesh files organized by robot type
- Easy to add new environments

---

## Import Patterns

### From Core
```python
from modern_envs.core import ModernMujocoEnv, GoalEnv, GymGoalEnvWrapper
```

### From Wrappers
```python
from modern_envs.wrappers import DiscretizedActionEnv
```

### From Envs
```python
from modern_envs.envs import SawyerPushGoalEnv, LunarEnv
```

### From Utils
```python
from modern_envs.utils import Serializable, MultitaskEnv
```

### Factory Pattern (Recommended)
```python
from modern_envs import create_env  # Easiest!
env = create_env('pusher')
```

---

## Environment Wrapper Stack

Complete transformation pipeline for HDM:

```
┌─────────────────────────────────────────┐
│  HDM Algorithm                          │
└──────────────┬──────────────────────────┘
               │
    ┌──────────▼──────────┐
    │ DiscretizedActionEnv│   modern_envs/wrappers/
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ GymGoalEnvWrapper   │   modern_envs/core/
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ SawyerPushGoalEnv   │   modern_envs/envs/
    └──────────┬──────────┘
               │
┌──────────────▼───────────────────┐
│ SawyerPushAndReachXYEnvModern    │   modern_envs/envs/
│  (uses ModernMujocoEnv)          │   modern_envs/core/
└──────────────┬───────────────────┘
               │
        ┌──────▼──────┐
        │ Modern      │
        │ MuJoCo      │
        │ Physics     │
        └─────────────┘
```

---

## Testing

### Run All Tests
```bash
# Test base MuJoCo class
python modern_envs/tests/test_base.py

# Test SawyerPush
python modern_envs/tests/test_sawyer_push.py

# Test Lunar Lander
python modern_envs/tests/test_lunar.py

# Test HDM integration
python modern_envs/tests/test_hdm_integration.py
```

---

## Adding New Environments

### For MuJoCo Environments

1. Create XML model in `assets/`
2. Copy mesh files if needed
3. Create env file in `envs/` (inherit from `ModernMujocoEnv`)
4. Add to `envs/__init__.py`
5. Add to `__init__.py` factory functions
6. Create test in `tests/`

### For Non-MuJoCo Environments

1. Create env file in `envs/` (inherit from `GoalEnv`)
2. Add to `envs/__init__.py`
3. Add to `__init__.py` factory functions
4. Create test in `tests/`

---

## Benefits of New Structure

✅ **Clearer organization** - Easy to find what you need  
✅ **Better modularity** - Reusable components  
✅ **Easier testing** - Tests separated from code  
✅ **Simpler imports** - Logical import hierarchy  
✅ **Better docs** - All documentation together  
✅ **Scalable** - Easy to add new environments  

---

**Last Updated:** 2025-11-27  
**Status:** ✅ Reorganization complete, all tests passing

