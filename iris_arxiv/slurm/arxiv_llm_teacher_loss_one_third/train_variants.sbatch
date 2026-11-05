#!/bin/bash

#SBATCH --job-name=iris-one-third-train
#SBATCH --account=engelhardlab
#SBATCH --partition=engelhardlab-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/work/YOUR_NETID/iris_arxiv/logs/one-third-train-%A_%a.out
#SBATCH --error=/work/YOUR_NETID/iris_arxiv/logs/one-third-train-%A_%a.err

set -eo pipefail

module purge
module load Anaconda3/2024.02
module load CUDA/12.4
source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh
conda activate /work/YOUR_NETID/iris_arxiv/conda/envs/iris-dcc-v2

export HF_HOME=/work/YOUR_NETID/iris_arxiv/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-8}"

IRIS_PYTHON=/work/YOUR_NETID/iris_arxiv/conda/envs/iris-dcc-v2/bin/python
EXPERIMENT_CONFIG=experiments/arxiv_llm_teacher_loss/config.one_third.toml
IRIS_CONFIG=config.arxiv.dcc.train-one-third.toml
VARIANTS=(baseline shuffled_w005 teacher_w005)
VARIANT="${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"

mkdir -p \
    /work/YOUR_NETID/iris_arxiv/logs \
    /work/YOUR_NETID/iris_arxiv/checkpoints/teacher_loss_v1_one_third \
    /work/YOUR_NETID/iris_arxiv/results/teacher_loss_v1_one_third
cd /hpc/home/YOUR_NETID/iris_arxiv

echo "Host: $(hostname)"
echo "Variant: ${VARIANT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

"${IRIS_PYTHON}" experiments/arxiv_llm_teacher_loss/run_training.py \
    --experiment-config "${EXPERIMENT_CONFIG}" \
    --iris-config "${IRIS_CONFIG}" \
    --variant "${VARIANT}"

echo "One-third training completed: ${VARIANT}"
