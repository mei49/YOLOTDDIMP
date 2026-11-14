"""Dataset preparation: zip or folder -> fresh data yaml with local paths.

Supported layouts (under the dataset root, possibly one wrapper folder deep):
  A) images/train, images/val|valid, images/test   (+ labels/<split>)
  B) train/images, val|valid/images, test/images   (+ <split>/labels)

The data.yaml shipped inside TD.zip hard-codes /content/TD/... (Colab), so only its class
names are used; paths are always rewritten.
"""

import zipfile
from pathlib import Path

import yaml

SPLITS = {"train": ("train",), "val": ("val", "valid", "validation"), "test": ("test",)}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _extract(zip_path: Path, out_root: Path) -> Path:
    dest = out_root / zip_path.stem
    marker = dest / ".extracted"
    if not marker.exists():
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            for m in z.infolist():  # refuse path traversal
                target = (dest / m.filename).resolve()
                if not str(target).startswith(str(dest.resolve())):
                    raise ValueError(f"unsafe path in zip: {m.filename}")
            z.extractall(dest)
        marker.touch()
    return dest


def _find_root(base: Path):
    """Return (root, layout) where layout is 'A' or 'B'. Searches base and up to 2 levels below."""
    cands = [base] + [p for p in base.glob("*") if p.is_dir()] + [p for p in base.glob("*/*") if p.is_dir()]
    for root in cands:
        if root.name in ("__MACOSX",) or "__MACOSX" in root.parts:
            continue
        if (root / "images" / "train").is_dir():
            return root, "A"
        if (root / "train" / "images").is_dir():
            return root, "B"
    raise FileNotFoundError(f"no images/train or train/images folder found under {base}")


def _split_dir(root: Path, layout: str, split: str):
    for name in SPLITS[split]:
        d = root / "images" / name if layout == "A" else root / name / "images"
        if d.is_dir():
            return d
    return None


def _names(base: Path, root: Path):
    for y in list(root.glob("*.yaml")) + list(base.rglob("data.yaml")) + list(base.rglob("*.yaml")):
        if "__MACOSX" in y.parts:
            continue
        try:
            d = yaml.safe_load(y.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and d.get("names"):
            n = d["names"]
            return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n), y
    raise FileNotFoundError(f"no yaml with `names` found in {base}")


def prepare(src="data/TD.zip", out_dir="data", yaml_name=None, verbose=True):
    """Prepare a dataset and return the path of a fresh data yaml with absolute local paths."""
    src, out_dir = Path(src), Path(out_dir)
    if not src.exists():
        raise FileNotFoundError(f"dataset not found: {src.resolve()} (put TD.zip in data/ or set DATA)")
    base = _extract(src, out_dir) if src.suffix.lower() == ".zip" else src
    root, layout = _find_root(base)
    names, names_src = _names(base, root)

    cfg = {"path": str(root.resolve())}
    counts = {}
    for split in SPLITS:
        d = _split_dir(root, layout, split)
        if d is not None:
            cfg[split] = str(d.relative_to(root))
            counts[split] = sum(1 for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)
    if "train" not in cfg or "val" not in cfg:
        raise FileNotFoundError(f"dataset at {root} needs at least train and val splits, found {list(counts)}")
    cfg["nc"] = len(names)
    cfg["names"] = dict(enumerate(names))

    out = out_dir / (yaml_name or f"{base.name}.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    if verbose:
        print(f"dataset root : {root}  (layout {'images/<split>' if layout == 'A' else '<split>/images'})")
        print(f"class names  : {names}  (from {names_src.name})")
        print(f"images       : {counts}")
        print(f"wrote        : {out.resolve()}")
    return out.resolve()


def make_synthetic(out="data/synth", n_train=40, n_val=10, n_test=10, nc=3, imgsz=160, layout="B", seed=0):
    """Tiny random-box YOLO dataset for pipeline tests (no real signal)."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    out = Path(out)
    for split, n in (("train", n_train), ("valid", n_val), ("test", n_test)):
        idir = out / ("images" if layout == "A" else split) / (split if layout == "A" else "images")
        ldir = out / ("labels" if layout == "A" else split) / (split if layout == "A" else "labels")
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = rng.integers(60, 120, (imgsz, imgsz, 3), dtype=np.uint8)
            lines = []
            for _ in range(rng.integers(1, 4)):
                c = int(rng.integers(nc))
                w, h = rng.uniform(0.15, 0.4, 2)
                x, y = rng.uniform(w / 2, 1 - w / 2), rng.uniform(h / 2, 1 - h / 2)
                x0, y0, x1, y1 = int((x - w / 2) * imgsz), int((y - h / 2) * imgsz), int((x + w / 2) * imgsz), int((y + h / 2) * imgsz)
                img[y0:y1, x0:x1] = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][c % 3]
                lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            Image.fromarray(img).save(idir / f"{split}_{i}.jpg")
            (ldir / f"{split}_{i}.txt").write_text("\n".join(lines) + "\n")
    # mimic TD.zip: a data.yaml with Colab paths that must be ignored
    (out / "data.yaml").write_text(
        yaml.safe_dump({"train": "/content/TD/train/images", "val": "/content/TD/valid/images", "nc": nc, "names": [f"cls{i}" for i in range(nc)]})
    )
    return out
