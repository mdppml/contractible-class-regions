This repository is associated with the paper “Empirical Evidence for Simply Connected Decision Regions in Image Classifiers.”

Given four images assigned the same predicted label by a classifier, the method constructs a quad-mesh surface spanning the loop formed by the four images. The surface is recursively refined, checked by grid sampling, and repaired using local decision-boundary information when necessary.

## 📂 Repository Structure

- [**simply_connected.py**](simply_connected.py)  
  Main implementation of the label-preserving quad filling procedure.

- [**simply_connected_slurm.sh**](simply_connected_slurm.sh)  
  SLURM template for running the method on an HPC cluster.

- [**imagenet_quads/**](imagenet_quads/)  
  Input directory containing model-specific four-image loops.

- [**results/**](results/)  
  Output directory for final `.txt` result files.

- [**checkpoints/**](checkpoints/)  
  Optional checkpoint directory for intermediate mesh states.

- [**rlogs/**](rlogs/)  
  SLURM output and error logs.

---

## 🧭 Pick a Workflow

- **Run one quad locally:** use `simply_connected.py` directly.
- **Run many quads on a cluster:** use `simply_connected_slurm.sh`.
- **Reproduce the main experiment:** run the SLURM array for each supported model.
- **Run ablations:** change the gray-RMS stopping threshold with `--stop-diameter-gray-rms`.

---

## ⚙️ Requirements

The code requires Python with PyTorch, torchvision, and Pillow.

```bash
pip install torch torchvision pillow
```

The experiments in the paper were run on GPU-enabled nodes. CUDA is strongly recommended.

---

## 🖼️ Input Format

The code expects four images per loop, named using the quad id and a letter:

```text
0001a.jpeg
0001b.jpeg
0001c.jpeg
0001d.jpeg
0002a.jpeg
0002b.jpeg
0002c.jpeg
0002d.jpeg
```

The expected folder layout is:

```text
imagenet_quads/
  resnet50/
    0001a.jpeg
    0001b.jpeg
    0001c.jpeg
    0001d.jpeg
  densenet121/
  efficientnet_b0/
  convnext_tiny/
  vit_b_16/
  swin_t/
```

Each set of four images should be assigned the same predicted label by the corresponding model.

---

## 🚀 Running a Single Quad

```bash
python simply_connected.py \
  --model-name resnet50 \
  --quad-id 1 \
  --quad-image-dir imagenet_quads/resnet50 \
  --results-dir results/resnet50 \
  --checkpoint-dir checkpoints/resnet50 \
  --expected-quads 1000 \
  --filename-width 4 \
  --stop-diameter-gray-rms 0.5 \
  --max-grid-steps 512
```

The final result is written to:

```text
results/resnet50/quad1.txt
```

---

## 🧪 Supported Models

The following torchvision ImageNet models are supported:

```text
resnet50
densenet121
efficientnet_b0
convnext_tiny
vit_b_16
swin_t
```

---

## 🖥️ Running on SLURM

Before using the SLURM script, replace the template values inside `simply_connected_slurm.sh`.

Required replacements:

```text
PARTITION_NAME
PATH_TO_REPOSITORY
PATH_TO_CONDA_ENV
```

Then submit one model at a time:

```bash
sbatch simply_connected_slurm.sh 1
sbatch simply_connected_slurm.sh 2
sbatch simply_connected_slurm.sh 3
sbatch simply_connected_slurm.sh 4
sbatch simply_connected_slurm.sh 5
sbatch simply_connected_slurm.sh 6
```

Model indices:

```text
1 = resnet50
2 = densenet121
3 = efficientnet_b0
4 = convnext_tiny
5 = vit_b_16
6 = swin_t
```

---

## 📊 Outputs

Each successful run writes a text result file containing:

```text
final success or failure status
target predicted label
root quad acceptance status
number of levels processed
number of leaf quads
number of vertices
number of grid samples evaluated
number of repair attempts
DeepFool repair statistics
constructed surface area
boundary-matched Coons reference area
constructed-to-Coons area ratio
runtime statistics
area coverage table by depth
```

The main scientific outputs are:

- success rate across loops and models
- root-level acceptance rate
- parameter-domain coverage by refinement level
- final mesh complexity
- constructed-to-Coons area ratio
- DeepFool repair statistics
- runtime and computational cost
- gray-RMS threshold ablations

---

## 📏 Main Parameters

- `--stop-diameter-gray-rms`  
  Gray-level RMS diameter threshold below which a quad is accepted by resolution.

- `--max-grid-steps`  
  Maximum grid resolution used when checking a quad.

- `--deepfool-iter`  
  Maximum number of DeepFool-style repair iterations.

- `--repair-batch-size`  
  Batch size for repairing off-label vertices.

- `--sample-batch-size`  
  Batch size for classifier evaluations during grid checking.

- `--boundary-reference-grid`  
  Grid size used to estimate the boundary-matched Coons reference area.

---

## 🔁 RMS Threshold Ablation

To run the same quad set with a stricter or looser gray-RMS threshold, change:

```bash
--stop-diameter-gray-rms 0.5
```

For example:

```bash
--stop-diameter-gray-rms 1.0
--stop-diameter-gray-rms 0.25
--stop-diameter-gray-rms 0.125
```

---
## Practical Notes

The code uses torchvision pretrained ImageNet weights. Model-specific preprocessing is loaded automatically from torchvision.

For each model, the quad image folder should be generated separately, because different models may assign different predicted labels to the same ImageNet image.

Large runs can occasionally fail for infrastructure reasons rather than algorithmic reasons. CUDA out-of-memory errors are usually resolved by reducing `--sample-batch-size`, `--grid-point-block`, `--label-batch-size`, or `--repair-batch-size`. For memory-limited GPUs, it may also help to lower `--max-grid-steps` or reduce the number of simultaneous SLURM array tasks.

If jobs are interrupted by the scheduler, increase the requested wall time or rerun the missing quad IDs. The script skips completed result files unless `--overwrite` is passed.

Algorithmic failures should be interpreted separately from infrastructure failures. These occur when the filling procedure cannot repair a newly introduced vertex. In such cases, the DeepFool repair parameters, such as `--deepfool-iter`, `--overshoot`, and `--ls-steps`, can be adjusted as described in the paper.

---

## 📝 Citation

```
```

---

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
