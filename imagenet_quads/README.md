# ImageNet Quad Inputs

This folder stores model-specific four-image loops used by the surface-filling code.

Expected layout:

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

Each group of four images should have the same predicted label under the corresponding model.

This directory is intentionally not populated in the repository. Generate the quad folders with `make_imagenet_quads.py` or provide your own images in the same format.
