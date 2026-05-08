#contractible-class-regions 

Code for the paper [**Empirical Evidence for Simply Connected Decision Regions in Image Classifiers**](https://arxiv.org/abs/2605.06380).

Given four images assigned the same predicted label by a classifier, this repository constructs a quad-mesh surface spanning the loop formed by the four images. The surface is recursively refined, checked by grid sampling, and repaired using local decision-boundary information when necessary.

---

## 📂 Repository Structure

- [**simply_connected.py**](simply_connected.py)  
  Main implementation of the label-preserving quad filling procedure.

- [**simply_connected_slurm.sh**](simply_connected_slurm.sh)  
  SLURM template for running the surface-filling method on an HPC cluster.

- [**make_imagenet_quads.py**](make_imagenet_quads.py)  
  Script for generating model-specific ImageNet quad sets.

- [**make_imagenet_quads_slurm.sh**](make_imagenet_quads_slurm.sh)  
  SLURM template for generating quad sets on an HPC cluster.

- [**requirements.txt**](requirements.txt)  
  Minimal Python package requirements.

- [**environment.yml**](environment.yml)  
  Optional conda environment specification.

- [**imagenet_quads/**](imagenet_quads/)  
  Input directory containing model-specific four-image loops. This directory is generated locally and should not be committed.

- [**results/**](results/)  
  Output directory for final `.txt` result files. This directory is generated locally and should not be committed.

- [**checkpoints/**](checkpoints/)  
  Optional checkpoint directory for intermediate mesh states. This directory is generated locally and should not be committed.

- [**rlogs/**](rlogs/)  
  SLURM output and error logs. This directory is generated locally and should not be committed.

---

## 🧭 Pick a Workflow

- **Generate ImageNet quads:** use `make_imagenet_quads.py` or `make_imagenet_quads_slurm.sh`.
- **Run one quad locally:** use `simply_connected.py` directly.
- **Run many quads on a cluster:** use `simply_connected_slurm.sh`.
- **Reproduce the main experiment:** generate quads for each supported model, then run the filling SLURM array for each model.
- **Run ablations:** change the gray-RMS stopping threshold with `--stop-diameter-gray-rms`.

The prose in this README uses UK English, but the command-line flag remains `--stop-diameter-gray-rms` to match the implementation.

---

## ⚙️ Requirements

The code requires Python with PyTorch, torchvision, Pillow, Hugging Face datasets, and Hugging Face Hub.

Using pip:

```bash
pip install -r requirements.txt
```

Using conda:

```bash
conda env create -f environment.yml
conda activate simply-connected
```

CUDA is strongly recommended. The experiments in the paper were run on GPU-enabled nodes.

---

## 🖼️ Input Format

The surface-filling code expects four images per loop, named using the quad id and a letter:

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

ImageNet images should not be committed to the repository. Generate them locally using the quad-generation script.

---

## 🧩 Generating ImageNet Quads

The quad-generation script streams ImageNet validation images, classifies them with a chosen torchvision model, and stores four disk-stable images per predicted label. Disk-stable means that the image is saved as JPEG, reloaded from disk, and still assigned the same predicted label by the same model.

To generate quads for one model locally:

```bash
python make_imagenet_quads.py \
  --model resnet50 \
  --output-dir imagenet_quads \
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
```

This writes the quad images to:

```text
imagenet_quads/resnet50/
```

The script also writes metadata files:

```text
metadata.csv
label_summary.csv
quads.json
run.json
```

The ImageNet dataset on Hugging Face may require authentication. You can provide a token through the environment:

```bash
export HF_TOKEN=your_huggingface_token
```

or directly with:

```bash
--token your_huggingface_token
```

If your dataset uses a different image column name, pass it with:

```bash
--image-column image
```

For the main experiment, generate quad folders separately for each supported model, because different models may assign different predicted labels to the same ImageNet image.

---

## 🖥️ Generating Quads on SLURM

Before using the quad-generation SLURM script, replace the template values inside `make_imagenet_quads_slurm.sh`.

Required replacements:

```text
PARTITION_NAME
PATH_TO_REPOSITORY
PATH_TO_CONDA_ENV
```

Then submit:

```bash
sbatch make_imagenet_quads_slurm.sh
```

The default SLURM array generates quads for:

```text
resnet50
densenet121
efficientnet_b0
convnext_tiny
vit_b_16
swin_t
```

The generated folders are written to:

```text
imagenet_quads/<model_name>/
```

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

## ✅ Quick Test

A small test run can be used to check that the repository, dependencies, ImageNet access, and model loading work correctly.

First generate a tiny quad set:

```bash
python make_imagenet_quads.py \
  --model resnet50 \
  --output-dir imagenet_quads \
  --dataset-name ILSVRC/imagenet-1k \
  --split validation \
  --num-labels 2 \
  --images-per-label 4 \
  --filename-width 4 \
  --jpeg-quality 100 \
  --jpeg-subsampling 0 \
  --batch-size 16 \
  --verify-batch-size 16 \
  --max-records 2000
```

Then run the filling procedure on the first generated quad:

```bash
python simply_connected.py \
  --model-name resnet50 \
  --quad-id 1 \
  --quad-image-dir imagenet_quads/resnet50 \
  --results-dir results/resnet50_quicktest \
  --checkpoint-dir checkpoints/resnet50_quicktest \
  --expected-quads 2 \
  --filename-width 4 \
  --stop-diameter-gray-rms 0.5 \
  --max-grid-steps 128 \
  --sample-batch-size 32 \
  --label-batch-size 32 \
  --repair-batch-size 4
```

This quick test is not intended to reproduce the paper results. It only checks that the pipeline runs end to end.

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

## 🖥️ Running Surface Filling on SLURM

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

Each run writes one result file:

```text
results/<model_name>/quad<quad_id>.txt
```

This file contains the final status of the run and the quantities needed to reproduce the paper’s aggregate analyses, including:

```text
success or failure status
target predicted label
root-quad acceptance status
number of refinement levels processed
number of final leaf quads
number of verified vertices
number of grid samples evaluated
number of repair attempts
DeepFool repair iteration statistics
constructed surface area
boundary-matched Coons reference area
constructed-to-Coons area ratio
runtime
area coverage table by refinement level
```

---

## 📏 Main Parameters

- `--stop-diameter-gray-rms`  
  Grey-level RMS diameter threshold below which a quad is accepted by resolution.

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

To run the same quad set with a stricter or looser grey-RMS threshold, change:

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

```bibtex
@misc{swaminathan2026empiricalevidencesimplyconnected,
  title={Empirical Evidence for Simply Connected Decision Regions in Image Classifiers},
  author={Swaminathan, Arjhun and Akgün, Mete},
  year={2026},
  eprint={2605.06380},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.06380}
}
---

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
