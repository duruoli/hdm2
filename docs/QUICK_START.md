# Quick Start: Using External Environments with HDM

## Installation (One Time)

```bash
# Create environment with all dependencies
conda env create -f environment.yml
conda activate hdm2

# Install hdm and modern_envs packages
pip install -e .

# Verify installation
python -c "import torch, gymnasium, metaworld, mujoco, hdm, modern_envs; print('✅ Setup complete!')"
```

**See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for detailed installation instructions.**

## Test Installation

```bash
# Quick test - should show no errors if installed correctly
python tests/test_external_envs_integration.py --env metaworld_push
python tests/test_external_envs_integration.py --env fetch_push
python tests/test_external_envs_integration.py --env shadow_hand_block
```

## Train HDM (Basic)

```bash
# Metaworld push task (easy)
python -m hdm --env_name metaworld_push --independent_policy

# Fetch robot task (easy)
python -m hdm --env_name fetch_push --independent_policy

# Point maze navigation (medium)
python -m hdm --env_name pointmaze_medium --independent_policy

# Shadow hand manipulation (hard)
python -m hdm --env_name shadow_hand_block --independent_policy
```

## Train HDM (Full Command)

```bash
python -m hdm \
    --env_name metaworld_push \
    --independent_policy \
    --n_cycles 40 \
    --n_initial_rollouts 200 \
    --n_test_rollouts 50 \
    --batch_size 256 \
    --buffer_size 1000000 \
    --future_p 0.85 \
    --hdm_q_coef 1.0 \
    --hdm_bc \
    --hdm_weights_to_indicator \
    --seed 0
```

## Use Example Scripts

```bash
# Metaworld push (easy)
python examples/train_metaworld_push.py

# Shadow hand manipulation (hard)
python examples/train_hand_manipulate_pen.py
```

## Available Environments (28+ Confirmed Working)

### Metaworld (10 tasks) ✅ All Working
- `metaworld_push` ⭐ (recommended start)
- `metaworld_reach`
- `metaworld_door_open`
- `metaworld_door_close`
- `metaworld_pick_place`
- `metaworld_button_press`
- `metaworld_drawer_open`
- `metaworld_drawer_close`
- `metaworld_window_open`
- `metaworld_window_close`

### Gymnasium-Robotics 

**Fetch Robot (4 tasks)** ✅ All Working - Recommended
- `fetch_reach`
- `fetch_push` ⭐ (most stable)
- `fetch_slide`
- `fetch_pick_place`

**Point Maze Navigation (4 tasks)** ✅ All Working
- `pointmaze_open`
- `pointmaze_umaze`
- `pointmaze_medium` ⭐
- `pointmaze_large`

**Shadow Hand Manipulation (6+ tasks)** ✅ Working
- `shadow_hand_block` ⭐ (block manipulation)
- `shadow_hand_block_rotate`
- `shadow_hand_egg`
- `shadow_hand_egg_rotate`
- `shadow_hand_pen`
- `shadow_hand_pen_rotate`
- **Note**: Requires MuJoCo 3.2.0+ (see below if you get `apirate` errors)

**Adroit Hand Manipulation (4 tasks)** ✅ Working
- `adroit_hand_door` / `hand_door` ⭐ (door opening)
- `adroit_hand_hammer` / `hand_hammer`
- `adroit_hand_pen` / `hand_pen`
- `adroit_hand_relocate` / `hand_relocate`
- **Note**: Uses Box observation space with embedded goals (different from Shadow Hand)

## Programmatic Usage

```python
import modern_envs as envs

# Create any environment
env = envs.create_env('metaworld_push')  # or 'fetch_push', 'shadow_hand_block', etc.
env_params = envs.get_env_params('metaworld_push')

# Use with HDM
state = env.reset()
state, reward, done, info = env.step(action)
```

## Hyperparameter Tips

### Easy Tasks (push, reach)
- `n_cycles`: 40
- `n_initial_rollouts`: 200
- `independent_policy`: True

### Medium Tasks (fetch, maze)
- `n_cycles`: 60
- `n_initial_rollouts`: 300
- `independent_policy`: True

### Hard Tasks (shadow hand manipulation)
- `n_cycles`: 100-200
- `n_initial_rollouts`: 500-1000
- `independent_policy`: True
- Note: Shadow hand tasks require significantly more training

## Troubleshooting

**Error: Environment not found**
→ Install the required package (`pip install metaworld` or `pip install gymnasium-robotics`)

**Error: Shape mismatch**
→ Make sure you're using `--independent_policy` for large action spaces

**Error: `unrecognized attribute: 'apirate'` (Shadow Hand environments)**
→ This means old MuJoCo 2.x libraries are conflicting. See `MUJOCO_FIX_GUIDE.md` for the fix.
Quick fix: Comment out MuJoCo 2.x paths in `~/.zshrc` and install MuJoCo 3.2.0

**Training not converging**
→ Increase `n_initial_rollouts` and `n_cycles`

## Next Steps

- 📖 Read `EXTERNAL_ENVS_USAGE.md` for detailed documentation
- 🔧 Read `HDM_ENVIRONMENT_INTEGRATION_GUIDE.md` to add your own environments
- 📊 Read `INTEGRATION_SUMMARY.md` for technical details

---

**Ready to go!** Just pick an environment and run the command. 🚀

