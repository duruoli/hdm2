# VM Deployment and Performance Optimization Guide

## Quick Answer

**To deploy on your VM (32 CPUs, 1 GPU, 80GB RAM):**

1. **Clone and install:**
   ```bash
   git clone <your-repo-url>
   cd understanding-hindsight-goal-relabeling-supplementary
   conda env create -f environment.yml
   conda activate hdm2
   pip install -e .  # Install hdm and modern_envs packages
   ```

2. **Your code is ALREADY optimized for multi-core and GPU!** ✅
   - GPU will be used automatically if available
   - Multi-core CPU parallelization is built-in
   - See sections below for maximizing performance

---

## Installation on VM

### Prerequisites
```bash
# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    git \
    wget \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf
```

### Step 1: Install Conda (if not already installed)
```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh', 'Miniconda3-latest-Linux-x86_64.sh')"
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
conda init
# Restart shell or source ~/.bashrc
```

### Step 2: Clone Repository
```bash
git clone <your-repo-url>
cd understanding-hindsight-goal-relabeling-supplementary
```

### Step 3: Create Environment
```bash
# Create environment with all dependencies
conda env create -f environment.yml

# Activate environment
conda activate hdm2

# Install hdm and modern_envs packages
pip install -e .

# Verify installation
python -c "import torch, numpy, gymnasium, metaworld, mujoco; print('✅ All imports successful!')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import hdm, modern_envs; print('✅ HDM packages installed!')"
```

### Step 4: (Optional) Install MPI for Multi-Node Training
```bash
# If you want to use multiple VMs or MPI-based parallelization
sudo apt-get install -y libopenmpi-dev openmpi-bin
pip install mpi4py
```

---

## Performance Optimization: Using Your VM Resources

### ✅ GPU Utilization (AUTOMATIC)

Your code **already uses GPU automatically**! No configuration needed.

**How it works:**
- File: `hdm/utils/torch_utils.py`
  ```python
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  if use_cuda:
      cudnn.benchmark = True
      print('Using CUDA ..')
  ```
- All PyTorch tensors are automatically placed on GPU when available
- Neural network training will use your GPU

**Verify GPU is being used:**
```bash
# Before training
nvidia-smi

# During training (in another terminal)
watch -n 1 nvidia-smi
```

You should see GPU utilization increase during training.

---

### ✅ Multi-Core CPU Parallelization (AUTOMATIC)

Your code **already uses multiple CPU cores** through multiple mechanisms:

#### 1. **Environment Parallelization with `n_workers`**

The code uses `SubprocVecEnv` to run multiple environments in parallel:

```bash
# Use 16 parallel workers (for 32 CPUs)
python -m hdm --env_name metaworld_push \
    --independent_policy \
    --n_workers 16 \
    --n_cycles 50
```

**How it works:**
- Each worker runs an independent environment in a separate process
- Rollout collection is parallelized across workers
- Recommended: Use **50-75% of your CPUs** (16-24 workers for 32 CPUs)

#### 2. **Thread-level Parallelism (AUTOMATIC)**

The code sets optimal threading for NumPy/PyTorch operations:

```python
# From hdm/__main__.py lines 197-203
n_threads = str(args.n_workers)
if args.n_workers < 12:
    n_threads = str(12)

os.environ['OMP_NUM_THREADS'] = n_threads
os.environ['MKL_NUM_THREADS'] = n_threads
os.environ['IN_MPI'] = n_threads
```

This ensures efficient use of CPU threads for linear algebra operations.

#### 3. **MPI-based Distributed Training (OPTIONAL)**

For even more parallelism, you can use MPI to run multiple training processes:

```bash
# Install MPI (if not done)
pip install mpi4py

# Run with 4 MPI processes (each with 4 workers = 16 total workers)
mpirun -n 4 python -m hdm \
    --env_name metaworld_push \
    --independent_policy \
    --n_workers 4 \
    --n_cycles 50
```

**How MPI parallelization works:**
- Each MPI process collects rollouts independently
- Gradients are synchronized across processes using `Allreduce`
- Network parameters are synchronized after each update
- See `hdm/utils/mpi_utils.py` for implementation

---

## Recommended Configuration for Your VM

### Configuration 1: Maximum Throughput (16 Workers)
```bash
python -m hdm \
    --env_name metaworld_push \
    --independent_policy \
    --n_workers 16 \
    --n_cycles 100 \
    --n_initial_rollouts 200 \
    --batch_size 256 \
    --buffer_size 1000000
```

**Why:**
- 16 workers utilize ~50% of CPUs, leaving room for GPU training
- Large batch size (256) utilizes GPU memory
- Large buffer size (1M) uses ~10-20GB RAM

### Configuration 2: MPI Parallelism (4 MPI × 8 Workers = 32 total)
```bash
mpirun -n 4 python -m hdm \
    --env_name fetch_push \
    --independent_policy \
    --n_workers 8 \
    --n_cycles 100 \
    --batch_size 512
```

**Why:**
- 4 MPI processes × 8 workers each = 32 parallel environments
- Larger batch size (512) for better GPU utilization
- Gradient averaging across 4 processes for more stable training

### Configuration 3: Memory-Intensive (Large Buffer)
```bash
python -m hdm \
    --env_name shadow_hand_block \
    --independent_policy \
    --n_workers 16 \
    --buffer_size 5000000 \
    --batch_size 512 \
    --n_cycles 200
```

**Why:**
- 5M buffer size uses your 80GB RAM effectively
- Larger batch and buffer improve sample efficiency

---

## Monitoring Performance

### GPU Utilization
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Expected: 60-95% GPU utilization during training
```

### CPU Utilization
```bash
# Real-time monitoring
htop

# Or
top

# Expected: 50-80% total CPU usage with n_workers=16
```

### Memory Usage
```bash
# Check RAM usage
free -h

# During training, expect:
# - ~2-5GB per worker process
# - ~10-30GB for replay buffer (depending on buffer_size)
# - Total: 30-60GB for typical runs
```

### Training Progress
```bash
# Logs are saved to experiments/{env_name}/{timestamp}/
tail -f experiments/metaworld_push/*/progress.csv
```

---

## Performance Benchmarks (Estimated)

Based on your VM specs (32 CPUs, 1 GPU, 80GB RAM):

| Configuration | Rollouts/sec | Time per Cycle | Speedup vs 1 CPU |
|--------------|--------------|----------------|------------------|
| 1 worker, no GPU | ~100 | ~5 min | 1× |
| 1 worker, GPU | ~150 | ~3.5 min | 1.5× |
| 16 workers, GPU | ~2000 | ~30 sec | 10× |
| MPI 4×8 workers, GPU | ~3500 | ~20 sec | 15× |

**Note:** Actual performance depends on environment complexity.

---

## Troubleshooting

### Error: "modern_envs not found" or "hdm not found"

**Cause:** The local packages haven't been installed

**Solution:**
```bash
# Make sure you're in the repository root
cd /path/to/understanding-hindsight-goal-relabeling-supplementary

# Install packages in development mode
pip install -e .

# Verify
python -c "import hdm, modern_envs; print('✅ Packages installed!')"
```

**For Singularity/Docker containers:**
If you're in a container, make sure the repository is mounted and accessible:
```bash
# Inside container
cd /path/to/mounted/repo
pip install -e .
```

### GPU Not Being Used

**Check if PyTorch sees GPU:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"
```

**If False:**
1. Install CUDA-enabled PyTorch:
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. Verify NVIDIA drivers:
   ```bash
   nvidia-smi
   ```

### Out of Memory (GPU)

**Reduce batch size:**
```bash
python -m hdm --env_name metaworld_push --batch_size 128  # Instead of 256
```

### Out of Memory (RAM)

**Reduce buffer size or workers:**
```bash
python -m hdm --env_name metaworld_push --buffer_size 500000 --n_workers 8
```

### MPI Not Working

**Test MPI installation:**
```bash
mpirun -n 2 python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.Get_rank()}')"
```

**Expected output:**
```
Rank 0
Rank 1
```

---

## Example: Full Training Run

```bash
# 1. Clone and setup (one-time)
git clone <your-repo-url>
cd understanding-hindsight-goal-relabeling-supplementary
conda env create -f environment.yml
conda activate hdm2
pip install -e .

# 2. Verify installation
python -c "import torch, hdm, modern_envs; print(f'CUDA: {torch.cuda.is_available()}')"

# 3. Run optimized training
python -m hdm \
    --env_name metaworld_push \
    --independent_policy \
    --n_workers 16 \
    --n_cycles 100 \
    --n_initial_rollouts 200 \
    --batch_size 256 \
    --buffer_size 1000000 \
    --seed 42

# 4. Monitor in another terminal
watch -n 1 nvidia-smi
# or
htop
```

---

## Summary: Is Your Code Ready?

✅ **YES! Your code is already optimized for:**

1. **GPU Training** - Automatic, no config needed
2. **Multi-core CPU** - Use `--n_workers 16-24`
3. **Large Memory** - Use `--buffer_size 5000000`
4. **Distributed Training** - Optional MPI support

**You just need to:**
1. Clone the repo
2. Install dependencies: `conda env create -f environment.yml`
3. Run with recommended settings above

**No additional code changes needed!** 🚀

---

## Quick Reference

```bash
# Basic training (uses GPU + multi-core automatically)
python -m hdm --env_name metaworld_push --independent_policy --n_workers 16

# With MPI (4 processes × 8 workers each)
mpirun -n 4 python -m hdm --env_name metaworld_push --independent_policy --n_workers 8

# Full-featured training
python -m hdm \
    --env_name shadow_hand_block \
    --independent_policy \
    --n_workers 16 \
    --n_cycles 200 \
    --batch_size 512 \
    --buffer_size 5000000
```

For more details, see:
- [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) - Installation guide
- [QUICK_START.md](QUICK_START.md) - Basic usage
- [EXTERNAL_ENVS_USAGE.md](EXTERNAL_ENVS_USAGE.md) - Environment-specific info

