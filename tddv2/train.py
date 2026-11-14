"""Training / evaluation helpers for the notebook (resume, OOM fallback, smoke-test timing)."""

import gc
import time
from pathlib import Path

import torch

from . import register

register(verbose=False)

from ultralytics import YOLO  # noqa: E402

from .build import load_backbone  # noqa: E402


def _is_oom(e):
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_state(run_dir):
    """'new' | 'partial' | 'done' for runs/<name>."""
    last = Path(run_dir) / "weights" / "last.pt"
    if not last.exists():
        return "new"
    ckpt = torch.load(last, map_location="cpu", weights_only=False)  # our own checkpoint
    return "done" if ckpt.get("epoch", -1) == -1 else "partial"


def train_or_resume(cfg, name, data, S, project):
    """Train cfg as runs/<name>; resume from last.pt if a previous run was interrupted.

    S: settings dict (imgsz, batch, batch_oom, epochs, patience, workers, amp, seed, device, pretrained).
    Falls back to batch=S['batch_oom'] on CUDA OOM (ultralytics also halves batch itself during epoch 1).
    """
    run_dir = Path(project) / name
    state = run_state(run_dir)
    if state == "done":
        print(f"[{name}] already finished -> {run_dir / 'weights' / 'best.pt'}")
        return run_dir

    common = dict(device=S["device"], workers=S["workers"])
    batch = S["batch"]
    for attempt in range(2):
        try:
            if run_state(run_dir) == "partial":
                print(f"[{name}] resuming from {run_dir / 'weights' / 'last.pt'} (batch={batch})")
                YOLO(run_dir / "weights" / "last.pt").train(resume=True, batch=batch, **common)
            else:
                model = YOLO(cfg)
                if S.get("pretrained"):
                    r = load_backbone(model, S["pretrained"])
                    print(f"[{name}] backbone init from {S['pretrained']}: {r['transferred_0_10']}/{r['total_0_10']} tensors (layers 0-10)")
                model.train(
                    data=str(data), imgsz=S["imgsz"], batch=batch, epochs=S["epochs"], patience=S["patience"],
                    amp=S["amp"], seed=S["seed"], deterministic=True, project=str(project), name=name,
                    exist_ok=True, plots=True, **common,
                )
            return run_dir
        except RuntimeError as e:
            if not _is_oom(e) or attempt or batch <= S["batch_oom"]:
                raise
            print(f"[{name}] CUDA OOM at batch={batch}; retrying with batch={S['batch_oom']}")
            batch = S["batch_oom"]
            _free()
    return run_dir


class EpochTimer:
    """Callbacks that time the train and validation phases of each epoch."""

    def __init__(self):
        self.train_s, self.val_s, self._t0, self._t1 = [], [], 0.0, 0.0

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_epoch_start(self, trainer):
        self._sync()
        self._t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer):
        self._sync()
        self._t1 = time.perf_counter()
        self.train_s.append(self._t1 - self._t0)

    def on_fit_epoch_end(self, trainer):
        self._sync()
        self.val_s.append(time.perf_counter() - self._t1)

    def attach(self, model):
        for ev in ("on_train_epoch_start", "on_train_epoch_end", "on_fit_epoch_end"):
            model.add_callback(ev, getattr(self, ev))
        return self


def smoke_test(cfg, name, data, S, project, fraction=0.2):
    """1 epoch on `fraction` of the train split; returns timing and the projected full-run hours.

    full epoch ~= train_time / fraction + val_time   (val always runs on the full val split)
    projected  = epochs * full epoch                  (upper bound: early stopping via patience can end sooner)
    """
    model = YOLO(cfg)
    if S.get("pretrained"):
        load_backbone(model, S["pretrained"])
    timer = EpochTimer().attach(model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    model.train(
        data=str(data), imgsz=S["imgsz"], batch=S["batch"], epochs=1, fraction=fraction, amp=S["amp"], seed=S["seed"],
        device=S["device"], workers=S["workers"], project=str(project), name=name, exist_ok=True, plots=False, val=True,
    )
    wall = time.perf_counter() - t
    epoch_s = timer.train_s[0] / fraction + timer.val_s[0]
    peak = torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else float("nan")
    out = dict(
        model=name, train_s=round(timer.train_s[0], 1), val_s=round(timer.val_s[0], 1), wall_s=round(wall, 1),
        est_epoch_s=round(epoch_s, 1), projected_h=round(S["epochs"] * epoch_s / 3600, 2), peak_vram_gb=round(peak, 2),
        batch_used=model.trainer.batch_size,
    )
    del model
    _free()
    return out


def evaluate(weights, data, S, split="test", project=None, name=None):
    """Validate best.pt on a split; returns (summary dict, per-class AP50 dict)."""
    m = YOLO(weights)
    r = m.val(data=str(data), split=split, imgsz=S["imgsz"], batch=S["batch"], device=S["device"], workers=S["workers"],
              plots=False, project=str(project) if project else None, name=name, exist_ok=True, verbose=False)
    b = r.box
    names = r.names
    per_class = {names[int(c)]: float(ap) for c, ap in zip(r.ap_class_index, b.ap50)}
    summary = dict(P=float(b.mp), R=float(b.mr), mAP50=float(b.map50), **{"mAP50-95": float(b.map)})
    del m
    _free()
    return summary, per_class
