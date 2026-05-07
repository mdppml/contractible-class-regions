#!/usr/bin/env bash
#SBATCH --job-name=surface_fill
#SBATCH --partition=PARTITION_NAME
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --array=1-1000%100
#SBATCH --output=rlogs/%x_%A_%a.out
#SBATCH --error=rlogs/%x_%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
MODEL_INDEX="${1:-}"
RMS_VALUE="${2:-0.5}"
CONDA_ENV="${CONDA_ENV:-CONDA_ENV_NAME_OR_PATH}"

if [ -z "$MODEL_INDEX" ]; then
  echo "Usage: sbatch simply_connected_slurm.sh MODEL_INDEX [RMS_VALUE]"
  echo "MODEL_INDEX: 1=resnet50 2=densenet121 3=efficientnet_b0 4=convnext_tiny 5=vit_b_16 6=swin_t"
  exit 1
fi

case "$MODEL_INDEX" in
  1) MODEL_NAME="resnet50" ;;
  2) MODEL_NAME="densenet121" ;;
  3) MODEL_NAME="efficientnet_b0" ;;
  4) MODEL_NAME="convnext_tiny" ;;
  5) MODEL_NAME="vit_b_16" ;;
  6) MODEL_NAME="swin_t" ;;
  *)
    echo "Unknown MODEL_INDEX=$MODEL_INDEX"
    echo "MODEL_INDEX: 1=resnet50 2=densenet121 3=efficientnet_b0 4=convnext_tiny 5=vit_b_16 6=swin_t"
    exit 1
    ;;
esac

case "$MODEL_NAME" in
  resnet50|densenet121|efficientnet_b0)
    SAMPLE_BATCH_SIZE=512
    LABEL_BATCH_SIZE=256
    GRID_POINT_BLOCK=256
    REPAIR_BATCH_SIZE=64
    ;;
  convnext_tiny)
    SAMPLE_BATCH_SIZE=384
    LABEL_BATCH_SIZE=192
    GRID_POINT_BLOCK=192
    REPAIR_BATCH_SIZE=32
    ;;
  vit_b_16|swin_t)
    SAMPLE_BATCH_SIZE=192
    LABEL_BATCH_SIZE=96
    GRID_POINT_BLOCK=96
    REPAIR_BATCH_SIZE=16
    ;;
esac

RMS_TAG="${RMS_VALUE//./_}"
QUAD_IMAGE_DIR="$REPO_DIR/imagenet_quads/$MODEL_NAME"
RESULTS_DIR="$REPO_DIR/results/$MODEL_NAME/rms_$RMS_TAG"
CHECKPOINT_DIR="$REPO_DIR/checkpoints/$MODEL_NAME/rms_$RMS_TAG"
RLOGS_DIR="$REPO_DIR/logs/$MODEL_NAME/rms_$RMS_TAG"

mkdir -p "$RESULTS_DIR" "$CHECKPOINT_DIR" "$RLOGS_DIR" "$REPO_DIR/logs"

exec > >(tee -a "$RLOGS_DIR/output_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt")
exec 2> >(tee -a "$RLOGS_DIR/error_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt" >&2)

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

ulimit -n 8192 || true

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCHINDUCTOR_DISABLE=1
export TORCH_COMPILE_DISABLE=1
unset CUDA_LAUNCH_BLOCKING

echo "REPO_DIR=$REPO_DIR"
echo "MODEL_INDEX=$MODEL_INDEX"
echo "MODEL_NAME=$MODEL_NAME"
echo "RMS_VALUE=$RMS_VALUE"
echo "QUAD_IMAGE_DIR=$QUAD_IMAGE_DIR"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "RLOGS_DIR=$RLOGS_DIR"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HOSTNAME=$(hostname)"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "SAMPLE_BATCH_SIZE=$SAMPLE_BATCH_SIZE"
echo "LABEL_BATCH_SIZE=$LABEL_BATCH_SIZE"
echo "GRID_POINT_BLOCK=$GRID_POINT_BLOCK"
echo "REPAIR_BATCH_SIZE=$REPAIR_BATCH_SIZE"

python -u "$REPO_DIR/simply_connected.py" \
  --repo-dir "$REPO_DIR" \
  --model-name "$MODEL_NAME" \
  --quad-id "$SLURM_ARRAY_TASK_ID" \
  --quad-image-dir "$QUAD_IMAGE_DIR" \
  --results-dir "$RESULTS_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --expected-quads 1000 \
  --filename-width 4 \
  --stop-diameter-gray-rms "$RMS_VALUE" \
  --max-grid-steps 512 \
  --sample-batch-size "$SAMPLE_BATCH_SIZE" \
  --label-batch-size "$LABEL_BATCH_SIZE" \
  --grid-point-block "$GRID_POINT_BLOCK" \
  --repair-batch-size "$REPAIR_BATCH_SIZE" \
  --max-wall-hours 22.75
