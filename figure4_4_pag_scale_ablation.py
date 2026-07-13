"""
Figure 4.4 — PAG Scale Ablation (Mode 3 — PAG Enhanced)
=========================================================
Plots Sharpness and Visual Diversity vs PAG scale
s ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0} and saves a dual-axis line chart:
    output/figures/figure4_4_pag_scale_ablation.png

Sidecar .txt format (written by app.py):
    Attention: PAG Enhanced (scale=2.5)   ← scale parsed from here
    Sharpness: <float>                    ← used directly if present
    Full Prompt: <text>

Filename pattern (used for mode detection if sidecar absent):
    {Item}_{Style}_PAG_Enhanced_{seed1}_{seed2}.png

Only Mode 3 (PAG Enhanced) images are considered.
Change ITEM_PRESET, then run:  python figure4_4_pag_scale_ablation.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# =============================================================================
# ★ ONLY CHANGE THESE LINES ★
# =============================================================================
ITEM_PRESET = 1
# 1 = Sword   2 = Bow   3 = Polearm   4 = Armor   5 = Shield

PAG_SCALES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# If True, use LPIPS perceptual diversity (pip install lpips).
# Falls back to pixel-std diversity automatically if not installed.
USE_LPIPS = True
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

_ITEM_MAP = {1: "Sword", 2: "Bow", 3: "Polearm", 4: "Armor", 5: "Shield"}
if ITEM_PRESET not in _ITEM_MAP:
    raise ValueError("ITEM_PRESET must be 1–5")

ITEM       = _ITEM_MAP[ITEM_PRESET]
GEN_DIR    = PROJECT_ROOT / "output" / "kd_gen"
GEN_PREFIX = f"{ITEM}_"

IMAGE_EXTENSIONS  = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
OUTPUT_DIR  = PROJECT_ROOT / "output" / "figures"
OUTPUT_PATH = OUTPUT_DIR / "figure4_4_pag_scale_ablation.png"
DIVERSITY_IMG_SIZE = 256


# =============================================================================
# Sidecar parsing
# =============================================================================

def _read_sidecar(img_path: Path) -> dict[str, str]:
    sidecar = img_path.with_suffix(".txt")
    data: dict[str, str] = {}
    if not sidecar.is_file():
        return data
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            data[k.strip()] = v.strip()
    return data


def _is_pag_mode3(meta: dict[str, str], stem: str) -> bool:
    """Return True if this image was generated with PAG Enhanced (Mode 3)."""
    attention = meta.get("Attention", "").lower()
    if attention:
        return "pag enhanced" in attention and "adaptive" not in attention
    # Fallback: filename tokens
    low = stem.lower()
    return "pag" in low and "enhanced" in low and "adaptive" not in low


def _pag_scale_from_sidecar(meta: dict[str, str]) -> Optional[float]:
    """
    Extract scale from 'Attention: PAG Enhanced (scale=2.5)'.
    Returns None if not found.
    """
    m = re.search(r"scale\s*=\s*([\d.]+)", meta.get("Attention", ""), re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _parse_sharpness(meta: dict[str, str]) -> Optional[float]:
    try:
        return float(meta.get("Sharpness", ""))
    except ValueError:
        return None


# =============================================================================
# Sharpness (computed when sidecar value absent)
# =============================================================================

def _sharpness_from_array(arr: np.ndarray) -> float:
    from scipy.ndimage import laplace
    gray = (0.2126*arr[:,:,0] + 0.7152*arr[:,:,1] + 0.0722*arr[:,:,2]).astype(np.float32)
    return float(laplace(gray).var())


def get_sharpness(p: Path, meta: dict[str, str]) -> float:
    pre = _parse_sharpness(meta)
    if pre is not None:
        return pre
    arr = np.asarray(Image.open(p).convert("RGB"))
    return _sharpness_from_array(arr)


# =============================================================================
# Diversity metrics
# =============================================================================

def pixel_std_diversity(paths: list[Path]) -> float:
    size = (DIVERSITY_IMG_SIZE, DIVERSITY_IMG_SIZE)
    arrays = [np.asarray(Image.open(p).convert("RGB").resize(size, Image.Resampling.LANCZOS),
                         dtype=np.float32) / 255.0 for p in paths]
    return float(np.stack(arrays, axis=0).std(axis=0).mean())


def lpips_diversity(paths: list[Path], device: torch.device) -> float:
    import lpips  # type: ignore
    loss_fn = lpips.LPIPS(net="alex").to(device)
    size = (DIVERSITY_IMG_SIZE, DIVERSITY_IMG_SIZE)
    imgs = []
    for p in paths:
        arr = np.asarray(Image.open(p).convert("RGB").resize(size, Image.Resampling.LANCZOS),
                         dtype=np.float32) / 127.5 - 1.0
        imgs.append(torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device))
    dists = []
    with torch.no_grad():
        for i in range(len(imgs)):
            for j in range(i+1, len(imgs)):
                dists.append(float(loss_fn(imgs[i], imgs[j]).item()))
    return float(np.mean(dists)) if dists else float("nan")


def compute_diversity(paths: list[Path], device: torch.device) -> float:
    if len(paths) < 2:
        return float("nan")
    if USE_LPIPS:
        try:
            return lpips_diversity(paths, device)
        except ImportError:
            print("  [info] lpips not installed — using pixel-std diversity instead")
    return pixel_std_diversity(paths)


# =============================================================================
# Data collection
# =============================================================================

def collect_by_pag_scale(gen_dir: Path, prefix: str, scales: list[float]
                          ) -> dict[float, list[Path]]:
    if not gen_dir.is_dir():
        raise FileNotFoundError(
            f"Generated folder not found: {gen_dir}\n"
            "Run app.py with Mode 3 at varying PAG scales first."
        )
    buckets: dict[float, list[Path]] = {s: [] for s in scales}
    no_scale_count = 0

    for p in sorted(gen_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not p.name.startswith(prefix):
            continue
        meta = _read_sidecar(p)
        if not _is_pag_mode3(meta, p.stem):
            continue
        scale = _pag_scale_from_sidecar(meta)
        if scale is None:
            no_scale_count += 1
            continue
        if scale not in buckets:
            continue
        buckets[scale].append(p)

    if no_scale_count:
        print(f"  [info] {no_scale_count} PAG Enhanced images had no scale in sidecar — skipped.")
    return buckets


# =============================================================================
# Per-scale statistics
# =============================================================================

def compute_scale_stats(buckets, device) -> dict[float, dict]:
    results = {}
    for scale in sorted(buckets.keys()):
        paths = buckets[scale]
        if not paths:
            print(f"  ⚠  PAG scale {scale}: no images found")
            results[scale] = {"sharpness_mean": float("nan"), "sharpness_std": float("nan"),
                              "diversity": float("nan"), "n": 0}
            continue

        sharp_vals = [get_sharpness(p, _read_sidecar(p)) for p in paths]
        diversity  = compute_diversity(paths, device)
        n = len(paths)
        results[scale] = {
            "sharpness_mean": float(np.mean(sharp_vals)),
            "sharpness_std" : float(np.std(sharp_vals, ddof=1)) if n > 1 else 0.0,
            "diversity"     : diversity,
            "n"             : n,
        }
        div_str = f"{diversity:.4f}" if not np.isnan(diversity) else "N/A"
        print(f"  PAG {scale}: n={n}  sharpness={results[scale]['sharpness_mean']:.2f}  diversity={div_str}")
    return results


# =============================================================================
# Figure
# =============================================================================

def build_figure(scales, stats, item, diversity_label) -> None:
    valid = [s for s in scales if stats[s]["n"] > 0]
    if not valid:
        print("⚠  No data to plot.")
        return

    sharp_m = [stats[s]["sharpness_mean"] for s in valid]
    sharp_s = [stats[s]["sharpness_std"]  for s in valid]
    divers  = [stats[s]["diversity"]       for s in valid]
    ns      = [stats[s]["n"]               for s in valid]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    c_sharp, c_div = "#e74c3c", "#3498db"

    line1 = ax1.errorbar(valid, sharp_m, yerr=sharp_s,
                         marker="o", color=c_sharp, linewidth=2, capsize=4,
                         label="Sharpness (Laplacian var) ↑")
    ax1.set_xlabel("PAG Scale (s)", fontsize=12)
    ax1.set_ylabel("Sharpness", color=c_sharp, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=c_sharp)

    ax2    = ax1.twinx()
    line2  = None
    if any(not np.isnan(d) for d in divers):
        vx = [s for s, d in zip(valid, divers) if not np.isnan(d)]
        vy = [d for d in divers if not np.isnan(d)]
        line2 = ax2.plot(vx, vy, marker="s", color=c_div, linewidth=2, linestyle="--",
                         label=f"{diversity_label} ↑")[0]
        ax2.set_ylabel(diversity_label, color=c_div, fontsize=11)
        ax2.tick_params(axis="y", labelcolor=c_div)

    for s, sm, n in zip(valid, sharp_m, ns):
        ax1.annotate(f"n={n}", xy=(s, sm), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8, color="#666666")

    ax1.set_xticks(valid)
    ax1.set_title(
        f"Figure 4.4 — PAG Scale Ablation · Mode 3 (PAG Enhanced) · {item}\n"
        "Sharpness and Visual Diversity vs PAG scale",
        fontsize=12, fontweight="bold",
    )
    ax1.grid(axis="y", alpha=0.3)
    ax1.grid(axis="x", alpha=0.2)

    lines  = [line1] + ([line2] if line2 else [])
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=9)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Figure saved → {OUTPUT_PATH}")


# =============================================================================
# Console summary
# =============================================================================

def print_summary(scales, stats, diversity_label):
    print(f"\n{'#'*66}")
    print(f"Figure 4.4 summary — PAG Scale Ablation ({ITEM}, Mode 3)")
    print('#'*66)
    print(f"{'PAG s':<8} {'N':>5} {'Sharpness (mean±std)':>26} {diversity_label:>20}")
    print("-"*66)
    for s in scales:
        r  = stats[s]
        n  = int(r["n"])
        sh = "N/A" if np.isnan(r["sharpness_mean"]) else f"{r['sharpness_mean']:.2f} ± {r['sharpness_std']:.2f}"
        dv = "N/A" if np.isnan(r["diversity"])       else f"{r['diversity']:.4f}"
        print(f"{s:<8.1f} {n:>5} {sh:>26} {dv:>20}")
    print("-"*66)
    print("Sharpness ↑ | Diversity ↑")


# =============================================================================
# Main
# =============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diversity_label = "LPIPS Diversity" if USE_LPIPS else "Pixel-Std Diversity"

    print(f"Item         : {ITEM}")
    print(f"PAG scales   : {PAG_SCALES}")
    print(f"Gen dir      : {GEN_DIR}  (prefix '{GEN_PREFIX}', Mode 3 only)")
    print(f"Diversity    : {diversity_label}")
    print(f"Device       : {device}")

    buckets = collect_by_pag_scale(GEN_DIR, GEN_PREFIX, PAG_SCALES)
    for s, paths in sorted(buckets.items()):
        print(f"  PAG {s}: {len(paths)} images found")

    if all(len(v) == 0 for v in buckets.values()):
        raise ValueError(
            "No PAG Enhanced (Mode 3) images found with a 'scale=X.X' in the sidecar.\n"
            "Generate Mode 3 images with varying PAG scale in app.py and ensure the\n"
            "sidecar 'Attention:' line includes '(scale=X.X)'."
        )

    print("\nComputing metrics per PAG scale...")
    stats = compute_scale_stats(buckets, device)
    print_summary(PAG_SCALES, stats, diversity_label)
    build_figure(PAG_SCALES, stats, ITEM, diversity_label)


if __name__ == "__main__":
    main()
