# Result Files

This folder stores the final `.txt` output files produced by `simply_connected.py`.

Expected layout:

```text
results/
  resnet50/
    rms_0_5/
      quad1.txt
      quad2.txt
```

Each result file contains the final success or failure status, mesh statistics, repair statistics, area ratios, runtime information, and the area coverage table by refinement level.

Generated result files are ignored by git.
