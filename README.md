# Hindsight Divergence Minimization

This repository provides an implementation of Hindsight Divergence Minimization (HDM),
as proposed in the paper submission,
*Understanding Hindsight Goal Relabeling Requires Rethinking Divergence Minimization*.

If you use this codebase, please cite the anonymous paper submission.

## Quick Start

### Environment Setup

**⚠️ Use the verified environment specifications, not the old versions listed below!**

```bash
# Create environment from verified specifications
conda env create -f environment.yml

# Activate environment
conda activate hdm2

# Verify installation
python -c "import torch, numpy, gymnasium, metaworld, mujoco; print('✅ Setup complete!')"
```

**Verified versions** (Python 3.10.19, PyTorch 2.9.1, MuJoCo 3.2.0, etc.)  
See [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md) for detailed setup instructions.

### Running Experiments

```bash
# Quick test with Metaworld
python -m hdm --env_name metaworld_push --independent_policy

# Test with Gymnasium-Robotics
python -m hdm --env_name fetch_push --independent_policy

# Or use example scripts
python examples/train_metaworld_push.py
```

See [`docs/QUICK_START.md`](docs/QUICK_START.md) for more examples and 28+ available environments.

### Documentation

- 📖 **[Quick Start Guide](docs/QUICK_START.md)** - Get running in 5 minutes
- 🔧 **[Environment Setup](docs/ENVIRONMENT_SETUP.md)** - Detailed installation with verified versions
- 🌍 **[External Environments Usage](docs/EXTERNAL_ENVS_USAGE.md)** - Metaworld, Fetch, Shadow Hand, etc.
- 🔍 **[Integration Summary](docs/INTEGRATION_SUMMARY.md)** - Technical details
- 🛠️ **[MuJoCo Fix Guide](docs/MUJOCO_FIX_GUIDE.md)** - Troubleshooting MuJoCo issues

### Legacy GCSL Dependencies (Optional)

The original GCSL environments are included in the `gcsl/` folder.  
For modern environments (Metaworld, Gymnasium-Robotics), you only need the main `environment.yml` setup.

The training scripts are provided in `scripts` folder. 

## Development Notes

The current repo structure looks like the following:

 - `hdm` (Contains our implementation)
   - `agent` (defines the interface for an RL agent and the neural networks)
   - `algo` (defines the steps for environment sampling and training loops)
   - `learn` (defines the optimization procedure)
   - `replay` (defines the replay buffer with hindsight relabeling functionalities)
   - `utils` (utility functions that allow the training code to run on multiple cpu cores and each with multiple threads)

Experiments are logged into an `experiment` folder when the scripts are launched.

## Acknowledgements

This implementation is partially based on the following repos:

 - [OpenAI baselines](https://github.com/openai/baselines)
 - [Goal-Conditioned Supervised Learning](https://github.com/dibyaghosh/gcsl)
 - [PyTorch implementation of HER](https://github.com/TianhongDai/hindsight-experience-replay)
 - [World Model as a Graph](https://github.com/LunjunZhang/world-model-as-a-graph)
