# TDD-YOLO v2

An improved version of **TDD-YOLO** (YOLO11n-based tomato disease detector; Chen et al., PLOS ONE 2026,
doi [10.1371/journal.pone.0334989](https://doi.org/10.1371/journal.pone.0334989), code
[LingShaQ/TDD-YOLOCode](https://github.com/LingShaQ/TDD-YOLOCode)). It trains with **stock pip ultralytics**
(no fork) from one Jupyter notebook, and the settings are sized for an RTX 3050 with 6 GB of VRAM.

## What's changed and why

The released code has two bugs. They are the reason for v2:

1. **`CBAM_cor.forward()` (CSAF) builds its joint channel+spatial MLP with `nn.Sequential(...)` on every call.**
   It is re-initialised randomly on each forward pass and never trained. Its width (`C + H*W`) depends on the input
   resolution, so the CSAF fusion is effectively random noise.
2. **`co_FEM.forward()` (CAFE) calls `branch2_2` twice and never uses `branch2_3`.** The dilated 3x3 in the
   second coordinate-attention branch is dead weight (the build check shows it gets no gradient).

| | baseline (`cfg/tdd-yolo.yaml`) | v2 (`cfg/tdd-yolo-v2.yaml`) |
|---|---|---|
| CSAF | `CBAM_cor` (original) | **`CSAF2`**: channel logits from an avg+max pooled MLP; spatial logits from a 7x7 conv over `[max_c, mean_c, mean_c(x * channel weights)]`; a softmax of the spatial map pools `x` into a descriptor that goes through a second, zero-initialised MLP and is added to the channel logits; `out = x*sigmoid(ch) + x*sigmoid(sp)`. All weights are trained and nothing depends on H or W. |
| CAFE | `co_FEM` (original) | **`CAFE2`**: both branches complete (`1x3 -> *a_h -> 3x1 -> *a_w -> dilated 3x3`, plus the mirrored branch) |
| upsample | `nn.Upsample(nearest)` | **`DySample(scale=2, groups=4)`**, a learned point-sampling upsampler (Liu et al., ICCV 2023) |
| MSFF | `Averge_FEM` / `BackAverge_FEM` | unchanged |
| size @640 | 2.616 M params, 6.61 GFLOPs | 2.727 M params, 6.65 GFLOPs |

The baseline yaml uses the same layer rows as the authors' `ultralytics/backbone/yolo_corfem_canaver_corcbam.yaml`.
The only difference is the scale header: `depth_multiple: 0.50`, `width_multiple: 0.25`, which equals their `n` scale.

## Layout

```
TDD_YOLO_v2.ipynb     notebook: env check, settings, dataset, build check, smoke test, train v2,
                      train baseline, test-split comparison, curves, sample predictions
cfg/tdd-yolo.yaml     baseline (authors' modules)
cfg/tdd-yolo-v2.yaml  v2
tddv2/orig/           cor_fem.py, cbam_cor.py, aver_fem.py copied unchanged from ultralytics/nn/ourDemo
tddv2/modules.py      CSAF2, CAFE2, DySample
tddv2/register.py     patches ultralytics.nn.tasks.parse_model so the yamls work with pip ultralytics
tddv2/data.py         zip/folder -> fresh data yaml with local paths
tddv2/build.py        params/GFLOPs, forward/backward, yolo11n.pt backbone-transfer check
tddv2/train.py        train/resume, OOM fallback, smoke-test timing, evaluation
```

`tddv2/register.py` re-compiles `parse_model` from `inspect.getsource` with two edits. It adds the
channel-changing classes (`co_FEM`, `Averge_FEM`, `BackAverge_FEM`, `CAFE2`) to `base_modules`, and adds a branch
that passes `ch[f]` as the first argument to the channel-preserving classes (`CBAM_cor`, `CSAF2`, `DySample`). If
either regex fails to match, it raises an error instead of building a wrong model. That's why ultralytics is pinned
in `requirements.txt`.

## Setup (Windows or Linux, NVIDIA GPU)

Use Python 3.10–3.12. Install the **CUDA build of PyTorch first**: a plain `pip install torch` on Windows gets the CPU build.

```bash
git clone https://github.com/mei49/YOLOTDDIMP.git && cd YOLOTDDIMP
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux: source .venv/bin/activate

# tested torch 2.15.0; cu132 wheels need NVIDIA driver >= 580
pip install torch==2.15.0 torchvision --index-url https://download.pytorch.org/whl/cu132
#   older driver? use: pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Put the dataset at `data/TD.zip`. The `data.yaml` inside it points at `/content/TD/...`. That's fine: the notebook
only reads the class names from it and writes `data/TD.yaml` with local paths. Then:

```bash
jupyter notebook TDD_YOLO_v2.ipynb
```

`yolo11n.pt` downloads automatically the first time. It initialises layers 0-10 (the YOLO11n backbone) of **both**
models. Set `pretrained=None` in the settings cell to train both from scratch.

## Settings (notebook section 2)

imgsz 640, batch 16 (drops to 8 on CUDA OOM; ultralytics also halves the batch itself during epoch 1), 150 epochs,
patience 40, workers 4, AMP on, seed 0, deterministic. Both models use identical settings.

On a 4 GB RTX 3050 Laptop, expect the batch to drop to 8. If Windows dataloader workers cause trouble, set `workers=2` or `0`.

## Workflow

1. **Smoke test** (section 5): trains each model for 1 epoch on 20 % of the train split, times it with callbacks, and prints
   the projected hours for the full runs on your GPU (`epochs x (train_time / 0.2 + val_time)`). That's an upper bound,
   because early stopping usually ends runs sooner. Results go to `runs/smoke_timing.csv`.
2. **Train v2, then the baseline** (sections 6–7). If a run is interrupted, re-run its cell: it resumes from
   `runs/<name>/weights/last.pt`. A finished run is skipped.
3. **Compare** (section 8): evaluates `best.pt` of both runs on the **test** split and writes `runs/comparison.csv`
   (P, R, mAP50, mAP50-95, params, GFLOPs, plus a `v2 - baseline` row) and `runs/per_class_ap50.csv`.

The paper's Tomato-Disease numbers for TDD-YOLO (P 0.822, R 0.694, mAP50 0.797, mAP50-95 0.620, quoted from the
text that goes with Table 5) are included for reference. The paper used its own split and 200 epochs, so the
like-for-like result is **v2 minus the re-trained baseline**.

## Status

Verified on a CPU machine (Apple M4, torch 2.15.0, ultralytics 8.4.213):
- both yamls build, with the param/GFLOP numbers above
- train-mode forward/backward works, and so does inference at a second resolution
- `yolo11n.pt` transfers 200/200 tensors into layers 0-10 and touches nothing else
- the whole notebook runs end to end on a small synthetic YOLO dataset zip
- interrupt-and-resume works

**Not run yet:** the GPU smoke test and the two full trainings. Accuracy results will be added only after both full runs finish.
