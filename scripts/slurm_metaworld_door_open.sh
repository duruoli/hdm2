#!/bin/bash

#SBATCH --account=p31777  ## YOUR ACCOUNT pXXXX or bXXXX

#SBATCH --partition=gengpu  ### GPU PARTITION for GPU jobs

#SBATCH --gres=gpu:a100:1  ## Request 1 A100 GPU (required for gengpu partition)

#SBATCH --nodes=1  ## how many computers do you need

#SBATCH --ntasks-per-node=4  ## CPUs per node (reduced since GPU does main compute)

#SBATCH --time=12:00:00  ## 12 hours (max for gengpu is 48:00:00)

#SBATCH --mem=40G  ## RAM per node (40GB should be sufficient for this task)

#SBATCH --job-name=metaworld_door_open  ## Job name for identification

#SBATCH --output=metaworld_door_open_%j.out  ## Output file with job ID

#SBATCH --error=metaworld_door_open_%j.err  ## Error file with job ID

#SBATCH --mail-type=ALL  ## Email alerts for job status

#SBATCH --mail-user=duruoli2024@u.northwestern.edu  ## Your email

# ============================================
# Environment Setup
# ============================================

# Purge all modules to start clean
module purge all

# Load your manually installed Miniconda (installed to $HOME/miniconda)
source $HOME/miniconda/bin/activate

# Activate the hdm2 conda environment
conda activate hdm2

# If hdm2 doesn't exist on Quest, first create it:
# conda env create -f environment.yml

# Print environment info for debugging
echo "============================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"
python --version
nvidia-smi
echo "============================================"

# ============================================
# Run Training: Metaworld Door Open (Easy Task)
# ============================================

# Your project directory on Quest
cd /home/dlf8982/AAA/hdm2

# Run HDM training on metaworld_door_open
python -m hdm \
    --env_name metaworld_door_open \
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
    --hdm_online_o2 \
    --use_dqn \
    --double_dqn \
    --targ_clip \
    --seed 0

echo "============================================"
echo "Job completed at: $(date)"
echo "============================================"
