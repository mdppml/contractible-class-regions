#!/bin/bash
#SBATCH --job-name=make_imagenet_quads
#SBATCH --partition=PARTITION_NAME
#SBATCH --time=12:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-5%3
#SBATCH --output=PATH_TO_REPOSITORY/rlogs/make_quads_%A_%a.txt
#SBATCH --error=PATH_TO_REPOSITORY/rlogs/make_quads_error_%A_%a.txt

REPO_DIR=PATH_TO_REPOSITORY
CONDA_ENV=PATH_TO_CONDA_ENV

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

ulimit -n 8192 || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$REPO_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$REPO_DIR/.cache/huggingface/datasets"

mkdir -p "$REPO_DIR/rlogs"
mkdir -p "$HF_HOME"
mkdir -p "$HF_DATASETS_CACHE"

MODELS=(
  resnet50
  densenet121
  efficientnet_b0
  convnext_tiny
  vit_b_16
  swin_t
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$MODEL" ]; then
  echo "No model found for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 1
fi

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HOSTNAME=$(hostname)"
echo "REPO_DIR=$REPO_DIR"
echo "MODEL=$MODEL"

python -u "$REPO_DIR/make_imagenet_quads.py" \
  --model "$MODEL" \
  --output-dir "$REPO_DIR/imagenet_quads" \
  --dataset-name ILSVRC/imagenet-1k \
  --split validation \
  --num-labels 1000 \
  --images-per-label 4 \
  --filename-width 4 \
  --jpeg-quality 100 \
  --jpeg-subsampling 0 \
  --batch-size 64 \
  --verify-batch-size 64 \
  --use-channels-last \
  --use-tf32
