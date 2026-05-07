# Empirical Surface Filling for Classifier Decision Regions

This repository contains the code used to construct label-preserving quad-mesh fillings of four-image loops in ImageNet classifier decision regions.

## Files

- `simply_connected.py`: main Python script for running one quad filling job.
- `simply_connected_slurm.sh`: SLURM array script for running many quads in parallel.
- `imagenet_quads/`: expected input folder containing four images per quad.
- `results/`: output folder for one text result file per quad.
- `checkpoints/`: optional checkpoint folder.
- `logs/`: SLURM and runtime logs.

## Required replacements

Before running on a cluster, replace these template values in `simply_connected_slurm.sh`.

| Template | Replace with |
|---|---|
| `PARTITION_NAME` | Your SLURM partition name, for example `gpu`, `day`, or `compute`. |
| `CONDA_ENV_NAME_OR_PATH` | Your conda environment name or full conda environment path. |
| `#SBATCH --time=23:00:00` | Your desired wall-time limit. |
| `#SBATCH --gres=gpu:1` | Your cluster's GPU resource syntax if different. |
| `#SBATCH --cpus-per-task=4` | Number of CPU cores per quad job. |
| `#SBATCH --mem=60G` | Memory per quad job. |
| `#SBATCH --array=1-1000%100` | Quad IDs to run and maximum concurrent jobs. |

The script assumes it is submitted from the repository root or that `REPO_DIR` points to the repository root.

## Input data layout

The Python script expects four images per quad, named with the letters `a`, `b`, `c`, and `d`. For model `resnet50` and quad ID `1`, the default expected paths are:

```text
imagenet_quads/resnet50/0001a.jpeg
imagenet_quads/resnet50/0001b.jpeg
imagenet_quads/resnet50/0001c.jpeg
imagenet_quads/resnet50/0001d.jpeg
```

Supported image extensions are `.jpeg`, `.jpg`, and `.png`. The script also accepts non-zero-padded names such as `1a.jpeg`.

For all six supported models, the recommended layout is:

```text
imagenet_quads/
  resnet50/
  densenet121/
  efficientnet_b0/
  convnext_tiny/
  vit_b_16/
  swin_t/
```

Each model directory should contain the model-specific quads, because different models may assign different labels to the same images.

## Supported models

The script currently supports torchvision pretrained weights for:

- `resnet50`
- `densenet121`
- `efficientnet_b0`
- `convnext_tiny`
- `vit_b_16`
- `swin_t`

The SLURM script uses a model index:

| Index | Model |
|---:|---|
| 1 | `resnet50` |
| 2 | `densenet121` |
| 3 | `efficientnet_b0` |
| 4 | `convnext_tiny` |
| 5 | `vit_b_16` |
| 6 | `swin_t` |

## Python environment

Install the main dependencies in your conda environment:

```bash
conda create -n surface-fill python=3.10 -y
conda activate surface-fill
pip install torch torchvision pillow
```

Install the PyTorch build appropriate for your CUDA version using the official PyTorch instructions.

## Running one quad locally

From the repository root:

```bash
python -u simply_connected.py \
  --model-name resnet50 \
  --quad-id 1 \
  --quad-image-dir imagenet_quads/resnet50 \
  --results-dir results/resnet50/rms_0_5 \
  --checkpoint-dir checkpoints/resnet50/rms_0_5 \
  --expected-quads 1000 \
  --filename-width 4 \
  --stop-diameter-gray-rms 0.5 \
  --max-grid-steps 512
```

## Running on SLURM

After replacing the template values, submit from the repository root:

```bash
sbatch simply_connected_slurm.sh 1
```

This runs the 1000-quad array for ResNet-50 with the default gray-RMS threshold `0.5`.

To run another threshold:

```bash
sbatch simply_connected_slurm.sh 1 0.25
```

To control where files are read and written, set `REPO_DIR` explicitly:

```bash
REPO_DIR=/path/to/repository sbatch simply_connected_slurm.sh 1 0.5
```

## Outputs

For model `resnet50` and RMS threshold `0.5`, outputs are written to:

```text
results/resnet50/rms_0_5/quad1.txt
checkpoints/resnet50/rms_0_5/
logs/resnet50/rms_0_5/
```

Each `quad*.txt` file contains the full stdout trace and final summary for that quad, including:

- final success or failure status
- target label
- root-quad acceptance diagnostic
- levels processed
- number of grid-checked quads
- number of leaf quads
- number of created and verified vertices
- number of repair attempts
- DeepFool repair statistics
- constructed surface area
- boundary-matched Coons reference area
- runtime breakdown
- per-level area coverage table

## Important parameters

| Parameter | Meaning |
|---|---|
| `--stop-diameter-gray-rms` | Diameter threshold for accepting sufficiently small quads. |
| `--max-grid-steps` | Maximum sampled grid side used for quad verification. |
| `--sample-batch-size` | Maximum number of sampled surface points evaluated per model batch. |
| `--label-batch-size` | Batch size for checking newly created vertices. |
| `--grid-point-block` | Number of grid points processed together when scanning quads. |
| `--repair-batch-size` | Number of off-label vertices repaired together. |
| `--deepfool-iter` | Maximum iterations for the targeted DeepFool-style repair step. |
| `--max-wall-hours` | Python-side wall-time guard inside each SLURM job. |

## Notes on reproducibility

The script uses `SLURM_ARRAY_TASK_ID` as the quad ID when `--quad-id` is not provided. If `--quad-id` is provided, it overrides the SLURM array ID.

The random seed is set as `seed + quad_id`.

The code uses torchvision's default pretrained weights and corresponding preprocessing transforms.

## Citation

If you use this code, cite the accompanying paper.
