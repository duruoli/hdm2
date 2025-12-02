# Documentation Index

Welcome to the HDM (Hindsight Divergence Minimization) documentation!

## 📚 Documentation Overview

### Getting Started (Start Here!)

1. **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)** ⭐ **START HERE**
   - Complete environment setup with **verified package versions**
   - Python 3.10.19, PyTorch 2.9.1, MuJoCo 3.2.0, etc.
   - Installation instructions for conda and pip
   - Troubleshooting common issues
   - **Use this, NOT the old README versions!**

2. **[QUICK_START.md](QUICK_START.md)** ⚡
   - Get running in 5 minutes
   - 28+ available environments (Metaworld, Fetch, Shadow Hand, etc.)
   - Basic training commands
   - Example scripts

### Detailed Guides

3. **[EXTERNAL_ENVS_USAGE.md](EXTERNAL_ENVS_USAGE.md)** 🌍
   - Comprehensive guide to external environments
   - Metaworld (10 tasks)
   - Gymnasium-Robotics (Fetch, Shadow Hand, Adroit, Mazes)
   - Environment-specific configuration
   - Advanced usage

4. **[VM_DEPLOYMENT_GUIDE.md](VM_DEPLOYMENT_GUIDE.md)** 🖥️ **NEW!**
   - Deploy on cloud VMs and servers
   - GPU and multi-core CPU optimization
   - Automatic parallelization (already built-in!)
   - MPI-based distributed training
   - Performance benchmarks and monitoring

5. **[INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)** 🔍
   - Technical details of how HDM integrates with environments
   - Architecture overview
   - Implementation details
   - Code structure

6. **[MUJOCO_FIX_GUIDE.md](MUJOCO_FIX_GUIDE.md)** 🛠️
   - Troubleshooting MuJoCo issues
   - Fixing 'apirate' errors (Shadow Hand)
   - Version conflict resolution
   - Platform-specific tips

### Legacy Documentation

7. **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** (in `modern_envs/docs/`)
   - How to migrate from old environments to new ones
   - API changes

## 🗂️ Project Structure

```
understanding-hindsight-goal-relabeling-supplementary/
├── README.md                    # Main project README (updated with new setup)
├── environment.yml              # ⭐ Conda environment (VERIFIED versions)
├── requirements.txt             # ⭐ Pip requirements (VERIFIED versions)
├── requirements-external-envs.txt  # Legacy external env requirements
│
├── docs/                        # 📖 All documentation
│   ├── INDEX.md                 # This file
│   ├── ENVIRONMENT_SETUP.md     # ⭐ START HERE - Setup guide
│   ├── QUICK_START.md           # Quick start guide
│   ├── EXTERNAL_ENVS_USAGE.md   # External environments guide
│   ├── VM_DEPLOYMENT_GUIDE.md   # VM deployment & performance optimization
│   ├── INTEGRATION_SUMMARY.md   # Technical integration details
│   └── MUJOCO_FIX_GUIDE.md      # MuJoCo troubleshooting
│
├── tests/                       # 🧪 All test files
│   ├── test_external_envs_integration.py
│   ├── test_gymnasium_metaworld.py
│   ├── test_hdm_training.py
│   ├── test_metaworld_hdm_training.py
│   ├── analyze_adroit_obs.py
│   └── README.md                # Test documentation
│
├── examples/                    # 💡 Example training scripts
│   ├── train_metaworld_push.py
│   └── train_hand_manipulate_pen.py
│
├── hdm/                         # 🧠 Core HDM implementation
│   ├── agent/                   # RL agent and neural networks
│   ├── algo/                    # Training algorithms
│   ├── learn/                   # Optimization procedures
│   ├── replay/                  # Replay buffer with hindsight relabeling
│   └── utils/                   # Utility functions
│
├── modern_envs/                 # 🌍 Modern environment wrappers
│   ├── envs/                    # Environment implementations
│   ├── wrappers/                # Goal environment wrappers
│   ├── core/                    # Base classes
│   └── tests/                   # Environment-specific tests
│
├── gcsl/                        # 📦 Legacy GCSL dependencies (optional)
├── experiments/                 # 📊 Training results and logs
└── scripts/                     # 🚀 Bash training scripts
```

## 🚀 Quick Navigation

**I want to...**

- **Set up my environment** → [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md)
- **Run my first experiment** → [`QUICK_START.md`](QUICK_START.md)
- **Deploy on a VM or server** → [`VM_DEPLOYMENT_GUIDE.md`](VM_DEPLOYMENT_GUIDE.md) ⭐
- **Use Metaworld/Fetch/Shadow Hand** → [`EXTERNAL_ENVS_USAGE.md`](EXTERNAL_ENVS_USAGE.md)
- **Fix MuJoCo errors** → [`MUJOCO_FIX_GUIDE.md`](MUJOCO_FIX_GUIDE.md)
- **Understand the codebase** → [`INTEGRATION_SUMMARY.md`](INTEGRATION_SUMMARY.md)
- **Run tests** → [`tests/README.md`](../tests/README.md)

## ⚠️ Important Notes

### Environment Versions

**DO NOT use the old versions from the original README!**

❌ **Old (Don't use):**
- Python 3.7.4
- PyTorch 1.10.0
- NumPy 1.19.1
- Gym 0.13.1
- mujoco_py 2.0.2.13

✅ **Current (Use these):**
- Python 3.10.19
- PyTorch 2.9.1
- NumPy 2.2.6
- Gymnasium 1.2.2 + Gym 0.26.2
- MuJoCo 3.2.0

See [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md) for complete verified versions.

### MuJoCo Version

MuJoCo **3.2.0 or higher** is REQUIRED for Shadow Hand environments.  
Using MuJoCo 2.x will cause `unrecognized attribute: 'apirate'` errors.

### Conda Environment

The active conda environment is **`hdm2`** (not `hdm`).

```bash
conda activate hdm2  # ✅ Correct
conda activate hdm   # ❌ Old environment (deleted)
```

## 📞 Getting Help

1. Check the relevant documentation above
2. Look at troubleshooting sections in each guide
3. Check test files for usage examples
4. See `examples/` for working training scripts

## 🎯 Recommended Learning Path

For new users, follow this order:

1. **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)** - Set up your environment
2. **[QUICK_START.md](QUICK_START.md)** - Run your first experiment
3. **[EXTERNAL_ENVS_USAGE.md](EXTERNAL_ENVS_USAGE.md)** - Learn about available environments
4. **[VM_DEPLOYMENT_GUIDE.md](VM_DEPLOYMENT_GUIDE.md)** - Deploy on powerful hardware (optional)
5. **[INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)** - Understand the implementation
6. **[MUJOCO_FIX_GUIDE.md](MUJOCO_FIX_GUIDE.md)** - Troubleshoot if needed

---

**Ready to start?** → Begin with [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) 🚀


