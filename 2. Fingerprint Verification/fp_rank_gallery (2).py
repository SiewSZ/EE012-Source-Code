#!/usr/bin/env python
# pyright: reportMissingImports=false, reportInvalidTypeForm=false
# -*- coding: utf-8 -*-

"""
Rank gallery hands by fingerprint similarity to a single probe folder.

Inputs are the per-image fingerprint crops produced by tools/fingerprint_pipeline.py,
e.g. for each image folder under vis_fp/ you have files like:
  finger_00_gabor.png / finger_00_skel.png / finger_00.png
  ...
  finger_04_gabor.png / finger_04_skel.png / finger_04.png

We compute an ORB feature match *fraction* per finger (higher = more similar),
aggregate across the 5 fingers (mean/median/min), then rank all gallery folders.

Usage (run from repo root):
  conda activate mmdet310
  python tools/fp_rank_gallery.py "vis_fp/<PROBE_FOLDER>" \
      --gallery_dir vis_fp --kind auto --agg mean \
      --features 2000 --ratio 0.90 --topk 10 \
      --csv rank.csv

Notes
- Paths with spaces are OK as long as you quote them ("...").
- --kind auto     : prefer *_gabor.png, fall back to *_skel.png, then raw *_*.png
- --ratio 0.90    : Lowe’s ratio test threshold (higher = stricter matches)
- --features 2000 : ORB keypoints per image (more = slower but better)
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Any

try:
    import cv2 as cv  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    cv = None  # type: ignore
    np = None  # type: ignore


# ----- config -----
FINGERS = [0, 1, 2, 3, 4]  # indices for the five fingers we expect


# ----- helpers -----
def pick_fp_path(folder: Path, idx: int, kind: str) -> Optional[Path]:
    """
    Decide which file to use for a given finger in a folder.
    kind='gabor' -> finger_XX_gabor.png
    kind='skel'  -> finger_XX_skel.png
    kind='auto'  -> try gabor, then skel, then raw finger_XX.png
    Returns None if nothing exists for that finger.
    """
    stem = f"finger_{idx:02d}"
    if kind == "gabor":
        candidates = [folder / f"{stem}_gabor.png"]
    elif kind == "skel":
        candidates = [folder / f"{stem}_skel.png"]
    else:  # auto
        candidates = [
            folder / f"{stem}_gabor.png",
            folder / f"{stem}_skel.png",
            folder / f"{stem}.png",
        ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_fp_image(folder: Path, idx: int, kind: str) -> Optional[Any]:
    """
    Load a fingerprint image for a finger index from a folder in GRAYSCALE.
    Returns None if not available or unreadable.
    """
    if cv is None:
        return None
    path = pick_fp_path(folder, idx, kind)
    if path is None:
        return None
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return None
    return img

def orb_score(
    p_img: Any,
    q_img: Any,
    nfeatures: int = 1000,
    ratio: float = 0.75,
) -> float:
    """
    Return the fraction of 'good' ORB matches (0..1).
    Safe against empty/degenerate descriptor sets and short knn pairs.
    """
    if cv is None:
        return 0.0

    # Init ORB
    try:
        orb = cv.ORB_create(nfeatures=int(nfeatures))
    except Exception:
        return 0.0

    # Keypoints & descriptors
    k1, d1 = orb.detectAndCompute(p_img, None)
    k2, d2 = orb.detectAndCompute(q_img, None)
    if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
        return 0.0

    # BFMatcher + Lowe's ratio test (k=2)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    pairs = bf.knnMatch(d1, d2, k=2)
    if not pairs:
        return 0.0

    good: List[Any] = []
    r = float(ratio)
    for p in pairs:
        # Some entries may have only 1 neighbor; skip those safely
        if p is None or len(p) < 2:
            continue
        m, n = p[0], p[1]
        if m is not None and n is not None and m.distance < r * n.distance:
            good.append(m)

    # Score = fraction of good matches
    return float(len(good)) / float(max(1, len(pairs)))




def aggregate(vals: Iterable[float], how: str = "mean") -> float:
    """
    Combine per-finger scores into one value.
    """
    arr = [v for v in vals if v is not None]
    if len(arr) == 0:
        return 0.0
    if how == "median":
        return float(statistics.median(arr))
    if how == "min":
        return float(min(arr))
    return float(statistics.mean(arr))  # default: mean


# ----- core ranking -----
def rank_gallery(
    probe_dir: Path,
    gallery_dir: Path,
    kind: str = "auto",
    agg: str = "mean",
    features: int = 2000,
    ratio: float = 0.90,
    topk: int = 10,
) -> List[Tuple[float, str]]:
    """
    For a single probe folder, compute similarity against every other folder
    under gallery_dir. Return a list of (score, folder_name) sorted desc.
    """
    # Preload probe images per finger once
    probe_imgs = {idx: load_fp_image(probe_dir, idx, kind=kind) for idx in FINGERS}

    # List candidate gallery folders (sibling dirs), excluding probe itself
    all_dirs = [p for p in sorted(Path(gallery_dir).iterdir()) if p.is_dir()]
    gallery = [g for g in all_dirs if g.name != probe_dir.name]
    if len(gallery) == 0:
        print("[warn] gallery is empty after excluding the probe.")
        return []

    results: List[Tuple[float, str]] = []

    for g in gallery:
        per_finger: List[float] = []
        for idx in FINGERS:
            p_img = probe_imgs[idx]
            q_img = load_fp_image(g, idx, kind=kind)
            if p_img is None or q_img is None:
                # missing data for this finger in either probe or gallery
                continue
            s = orb_score(p_img, q_img, nfeatures=features, ratio=ratio)
            per_finger.append(s)

        combined = aggregate(per_finger, how=agg)
        results.append((combined, g.name))

    results.sort(reverse=True, key=lambda x: x[0])
    return results[: topk if topk > 0 else len(results)]


# ----- CLI -----
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rank gallery hands by fingerprint similarity to a probe folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("probe", help="Path to probe folder (e.g., vis_fp/<ONE_FOLDER>)")
    p.add_argument("--gallery_dir", default="vis_fp", help="Parent dir of gallery folders")
    p.add_argument("--kind", choices=["auto", "gabor", "skel"], default="auto",
                   help="Which enhanced image to use per finger")
    p.add_argument("--agg", choices=["mean", "median", "min"], default="mean",
                   help="How to combine five finger scores")
    p.add_argument("--features", type=int, default=2000, help="ORB nfeatures per image")
    p.add_argument("--ratio", type=float, default=0.90, help="Lowe ratio test threshold")
    p.add_argument("--topk", type=int, default=10, help="How many results to print")
    p.add_argument("--csv", type=str, default="", help="Optional path to save CSV (rank,score,folder)")
    return p


def main() -> None:
    if cv is None:
        raise SystemExit(
            "[err] OpenCV (cv2) not importable in this interpreter.\n"
            "Activate your env first:  conda activate mmdet310\n"
        )

    ap = build_argparser()
    args = ap.parse_args()

    probe_dir = Path(args.probe)
    gallery_dir = Path(args.gallery_dir)

    if not probe_dir.exists() or not probe_dir.is_dir():
        raise SystemExit(f"[err] probe folder not found: {probe_dir}")
    if not gallery_dir.exists() or not gallery_dir.is_dir():
        raise SystemExit(f"[err] gallery_dir not found: {gallery_dir}")

    hits = rank_gallery(
        probe_dir=probe_dir,
        gallery_dir=gallery_dir,
        kind=args.kind,
        agg=args.agg,
        features=args.features,
        ratio=args.ratio,
        topk=args.topk,
    )

    # Print to stdout
    print(f"\n# probe: '{probe_dir.name}' | kind: {args.kind} | agg: {args.agg}")
    print(f"# features: {args.features} | ratio: {args.ratio:.2f} | topk: {args.topk}")
    print("rank,score,folder")
    for i, (score, name) in enumerate(hits, 1):
        print(f"{i},{score:.3f},{name}")

    # Optional CSV
    if args.csv:
        outp = Path(args.csv)
        with outp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["probe", probe_dir.name])
            w.writerow(["kind", args.kind])
            w.writerow(["agg", args.agg])
            w.writerow(["features", args.features])
            w.writerow(["ratio", args.ratio])
            w.writerow([])
            w.writerow(["rank", "score", "folder"])
            for i, (score, name) in enumerate(hits, 1):
                w.writerow([i, f"{score:.6f}", name])
        print(f"[ok] wrote CSV: {outp.resolve()}")


if __name__ == "__main__":
    main()

