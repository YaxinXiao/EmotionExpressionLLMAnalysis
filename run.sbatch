#!/bin/bash
#SBATCH --job-name=qwen-emotion-en
#SBATCH --account=alrodri
#SBATCH --partition=alrodri-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=qwen-emotion-en-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the Qwen_Video_Emotion directory}"
module load python/3.12.1
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export FORCE_QWENVL_VIDEO_READER=torchvision
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -u analyze.py "$@"
