"""Build / sanity-check helpers used by the notebook and tests."""

from pathlib import Path

import torch

from . import register

register(verbose=False)

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402

BACKBONE_LAYERS = range(0, 11)  # yolo11n layers 0-10 are identical in TDD-YOLO


def _layer(key):
    p = key.split(".")
    return int(p[1]) if len(p) > 1 and p[0] == "model" and p[1].isdigit() else None


def summarize(cfg, nc=80, imgsz=640):
    """Return (params, GFLOPs) for a yaml model."""
    m = YOLO(cfg).model
    if nc != 80:
        from ultralytics.nn.tasks import DetectionModel

        m = DetectionModel(cfg, nc=nc, verbose=False)
    params = sum(p.numel() for p in m.parameters())
    return params, get_flops(m, imgsz)


def forward_backward(cfg, nc=8, imgsz=256, device="cpu"):
    """One train-mode forward/backward on random input; checks every parameter gets a gradient."""
    from ultralytics.nn.tasks import DetectionModel

    m = DetectionModel(cfg, nc=nc, verbose=False).to(device).train()
    x = torch.randn(2, 3, imgsz, imgsz, device=device)
    out = m(x)
    outs = list(out.values()) if isinstance(out, dict) else list(out)
    outs = [t for o in outs for t in (o if isinstance(o, (list, tuple)) else [o])]
    sum(o.float().pow(2).mean() for o in outs).backward()
    no_grad = [n for n, p in m.named_parameters() if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    # a zero-initialised last layer (CSAF2.mlp_s[-1]) gets a gradient but the layer before it does not, at step 0
    no_grad = [n for n in no_grad if ".mlp_s.0." not in n]
    # resolution independence: a second size must work too
    m.eval()(torch.randn(1, 3, imgsz + 64, imgsz + 64, device=device))
    return [o.shape for o in outs], no_grad


def load_backbone(yolo, weights="yolo11n.pt"):
    """yolo.load(weights) and report which layers actually received pretrained tensors.

    Returns dict(transferred_0_10, total_0_10, changed_outside_0_10).
    """
    before = {k: v.clone() for k, v in yolo.model.state_dict().items()}
    yolo.load(weights)
    after = yolo.model.state_dict()
    changed = [k for k in after if after[k].dtype.is_floating_point and not torch.equal(before[k], after[k])]
    in_bb = [k for k in after if _layer(k) in BACKBONE_LAYERS and after[k].dtype.is_floating_point and "num_batches" not in k]
    return dict(
        transferred_0_10=sum(_layer(k) in BACKBONE_LAYERS for k in changed),
        total_0_10=len(in_bb),
        changed_outside_0_10=[k for k in changed if _layer(k) not in BACKBONE_LAYERS],
    )


def check_all(cfgs, weights=None, imgsz=640, device="cpu"):
    rows = []
    for cfg in cfgs:
        p, g = summarize(cfg, imgsz=imgsz)
        shapes, no_grad = forward_backward(cfg, device=device)
        row = dict(model=Path(cfg).stem, params=p, gflops=round(g, 2), fwd_bwd="ok", dead_params=sorted({n.rsplit(".", 3)[0] for n in no_grad}) or "-")
        if weights:
            r = load_backbone(YOLO(cfg), weights)
            row["backbone_transfer"] = f"{r['transferred_0_10']}/{r['total_0_10']} tensors in layers 0-10"
            row["other_layers_touched"] = len(r["changed_outside_0_10"])
        rows.append(row)
    return rows
