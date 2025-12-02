# Environment Setup Guide

This guide provides detailed instructions for setting up the HDM development environment with **verified package versions**.

## Quick Setup (Recommended)

### Option 1: Using Conda (Recommended for reproducibility)

```bash
# Create environment from verified specifications
conda env create -f environment.yml

# Activate the environment
conda activate hdm2

# Install hdm and modern_envs packages
pip install -e .

# Verify installation
python -c "import torch, numpy, gymnasium, metaworld, mujoco; print('✅ All core packages imported successfully!')"
python -c "import hdm, modern_envs; print('✅ HDM packages installed!')"
python --version  # Should show: Python 3.10.19
```

### Option 2: Using pip with existing Python 3.10

```bash
# Make sure you have Python 3.10.x installed
python --version  # Should be 3.10.x

# Install all dependencies
pip install -r requirements.txt

# Install hdm and modern_envs packages
pip install -e .

# Verify installation
python -c "import torch, numpy, gymnasium, metaworld, mujoco; print('✅ All core packages imported successfully!')"
python -c "import hdm, modern_envs; print('✅ HDM packages installed!')"
```

## Verified Environment Specifications

**System tested on:** macOS (Apple Silicon/arm64)  
**Date verified:** December 1, 2025

### Core Requirements

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10.19 | **Required** |
| PyTorch | 2.9.1 | Deep learning framework |
| NumPy | 2.2.6 | Numerical computing |
| SciPy | 1.15.3 | Scientific computing |

### RL Environment Packages

| Package | Version | Notes |
|---------|---------|-------|
| Gymnasium | 1.2.2 | Modern RL environment API |
| Gym | 0.26.2 | Legacy compatibility |
| Gymnasium-Robotics | 1.4.1 | Fetch, Hand, Maze environments |
| Metaworld | 3.0.0 | 10+ robotic manipulation tasks |
| MuJoCo | 3.2.0 | **≥3.2.0 required for Shadow Hand** |
| Box2D | 2.3.8 | 2D physics (Lunar Lander, etc.) |

### Important Version Notes

⚠️ **MuJoCo Version**: Must be **3.2.0 or higher** for Shadow Hand environments
- Older versions (2.x) will cause `unrecognized attribute: 'apirate'` errors
- See `MUJOCO_FIX_GUIDE.md` if you encounter MuJoCo version conflicts

⚠️ **Metaworld Version**: Using **3.0.0** (V3 API)
- This is different from older versions (≤0.4.x with V2 API)
- The V3 API has different environment naming and interfaces

⚠️ **Gymnasium vs Gym**: Both are installed for compatibility
- Gymnasium (1.2.2) is the modern, maintained version
- Gym (0.26.2) is legacy but still needed by some dependencies

## Testing Your Installation

### Quick Test

```bash
# Test that all imports work
python -c "import torch, numpy, gymnasium, metaworld, mujoco; print('✅ Success!')"

# Check versions
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
python -c "import mujoco; print(f'MuJoCo: {mujoco.__version__}')"
```

### Environment Integration Tests

```bash
# Test Metaworld environments
python tests/test_external_envs_integration.py --env metaworld_push

# Test Gymnasium-Robotics environments
python tests/test_external_envs_integration.py --env fetch_push
python tests/test_external_envs_integration.py --env shadow_hand_block

# Run comprehensive Metaworld tests
python tests/test_metaworld_hdm_training.py

# Run comprehensive Gymnasium tests
python tests/test_gymnasium_metaworld.py
```

### Quick Training Test

```bash
# Quick training run to verify everything works
python -m hdm --env_name metaworld_push --independent_policy --n_cycles 2 --n_initial_rollouts 10
```

If this completes without errors, your environment is properly set up! ✅

## Optional Dependencies

### For Distributed Training (MPI)

```bash
# Install Open MPI (macOS)
brew install open-mpi

# Install mpi4py
pip install mpi4py

# Test MPI
python -c "from mpi4py import MPI; print(f'MPI rank: {MPI.COMM_WORLD.Get_rank()}')"
```

### For Visualization and Logging

```bash
pip install tensorboard matplotlib pandas tqdm seaborn
```

### For Development

```bash
pip install pytest black flake8 mypy ipython jupyter
```

## Troubleshooting

### Issue: `unrecognized attribute: 'apirate'` (Shadow Hand environments)

**Cause:** MuJoCo 2.x libraries are conflicting with MuJoCo 3.2.0

**Solution:** See `MUJOCO_FIX_GUIDE.md` for detailed instructions

Quick fix:
1. Remove MuJoCo 2.x from your system
2. Ensure only MuJoCo 3.2.0+ is installed
3. Check your shell config files (~/.zshrc, ~/.bashrc) for old MuJoCo paths

### Issue: Import errors for `metaworld` or `gymnasium_robotics`

**Cause:** Packages not installed

**Solution:**
```bash
pip install metaworld>=3.0.0 gymnasium-robotics>=1.2.0
```

### Issue: NumPy version conflicts

**Cause:** Multiple packages depend on different NumPy versions

**Solution:**
```bash
# Force reinstall with specific version
pip install --force-reinstall numpy==2.2.6
```

### Issue: Rendering issues on Linux

**Cause:** Missing graphics libraries

**Solution (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev patchelf
```

### Issue: PyTorch installation issues

**Cause:** Platform-specific builds

**Solution:**
```bash
# For macOS Apple Silicon
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For Linux with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Updating from Old README Instructions

**⚠️ The README.md has outdated version information!**

Old README says:
- Python 3.7.4 ❌
- PyTorch 1.10.0 ❌
- NumPy 1.19.1 ❌
- Gym 0.13.1 ❌
- mujoco_py 2.0.2.13 ❌

**Use these verified versions instead** (from `environment.yml` and `requirements.txt`):
- Python 3.10.19 ✅
- PyTorch 2.9.1 ✅
- NumPy 2.2.6 ✅
- Gymnasium 1.2.2 + Gym 0.26.2 ✅
- MuJoCo 3.2.0 ✅

## Migration from Old Environment

If you have an old environment based on the README instructions:

```bash
# 1. Export your old environment (optional, for reference)
conda env export > old_environment.yml

# 2. Create new environment with verified versions
conda env create -f environment.yml

# 3. Activate new environment
conda activate hdm2

# 4. Test that everything works
python tests/test_external_envs_integration.py --env metaworld_push

# 5. (Optional) Remove old environment
conda env remove -n hdm  # or whatever your old env name was
```

## Platform-Specific Notes

### macOS (Apple Silicon)

- All packages work natively on Apple Silicon (arm64)
- PyTorch 2.9.1 has native Apple Silicon support
- MuJoCo 3.2.0 works well with Metal rendering

### Linux

- May need to install additional system libraries for rendering
- CUDA support available for PyTorch if you have an NVIDIA GPU

### Windows

- Not officially tested, but should work with appropriate dependencies
- May need Visual C++ redistributables

## Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. See `MUJOCO_FIX_GUIDE.md` for MuJoCo-specific issues
3. See `QUICK_START.md` for usage examples
4. Check `EXTERNAL_ENVS_USAGE.md` for environment-specific issues

## Summary

✅ Use `environment.yml` (conda) or `requirements.txt` (pip)  
✅ Python 3.10.19 required  
✅ MuJoCo 3.2.0+ required for Shadow Hand  
✅ Test with `python tests/test_external_envs_integration.py`  
❌ Don't use the old versions from README.md  

Your environment should be ready to train HDM agents! 🚀


