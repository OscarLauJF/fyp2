"""
Table 4.2 — Quantitative Benchmark Across Four Attention Modes
===============================================================
Computes Sharpness, SSIM, CLIP Score, and Inference Time for each mode
(mean ± std, up to 50 samples per mode) and prints a formatted table.

Sidecar .txt format (written by app.py):
    Attention: <Naive (Baseline) | Default (AttnProcessor2_0) |
                PAG Enhanced (scale=X.X) | Adaptive PAG + Progressive CFG (Proposed method)>
    Seed: <int>
    Time: <float>s
    Sharpness: <float>          ← pre-computed by app.py; used directly if present
    Full Prompt: <text>
    LoRA: <Yes | No>

Filename pattern (used as fallback mode detection):
    {Item}_{Style}_{Mode tokens}_{seed1}_{seed2}.png

For PAG Enhanced, only images at scale=2.5 are included in this table.
Change ITEM_PRESET, then run:  python table4_2_attention_mode_metrics.py
"""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import CLIPModel, CLIPProcessor

# =============================================================================
# ★ ONLY CHANGE THESE LINES ★
# =============================================================================
ITEM_PRESET = 1
# 1 = Sword   2 = Bow   3 = Polearm   4 = Armor   5 = Shield

MAX_SAMPLES_PER_MODE = 50

# PAG scale to use when selecting PAG Enhanced images for cross-mode comparison
PAG_SCALE_FOR_COMPARISON = 2.5

SAVE_CSV = True
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

_ITEM_MAP = {1: "Sword", 2: "Bow", 3: "Polearm", 4: "Armor", 5: "Shield"}
if ITEM_PRESET not in _ITEM_MAP:
    raise ValueError("ITEM_PRESET must be 1–5")

ITEM       = _ITEM_MAP[ITEM_PRESET]
GEN_DIR    = PROJECT_ROOT / "output" / "kd_gen"
GEN_PREFIX = f"{ITEM}_"
REAL_DIR   = PROJECT_ROOT / "data" / ITEM.lower()

CLIP_MODEL_ID    = "openai/clip-vit-base-patch32"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE       = 16
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FID_IMAGE_SIZE   = 299
SSIM_IMAGE_SIZE  = 512

MODE_NAMES = {
    1: "Mode 1 (Naive)",
    2: "Mode 2 (Default)",
    3: "Mode 3 (PAG Enhanced)",
    4: "Mode 4 (Adaptive PAG + Progressive CFG)",
}

OUTPUT_DIR = PROJECT_ROOT / "output" / "tables"
OUTPUT_CSV = OUTPUT_DIR / "table4_2_attention_mode_metrics.csv"

_MODE_ATTENTION_STRINGS: dict[int, list[str]] = {
    1: ["naive"],
    2: ["default"],
    3: ["pag enhanced"],
    4: ["adaptive pag"],
}


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


def _mode_from_sidecar(meta: dict[str, str]) -> Optional[int]:
    raw = meta.get("Attention", "").lower()
    if not raw:
        return None
    for mode_id, keywords in _MODE_ATTENTION_STRINGS.items():
        if all(kw in raw for kw in keywords):
            if mode_id == 3 and "adaptive" in raw:
                continue
            return mode_id
    return None


def _mode_from_filename(stem: str) -> Optional[int]:
    low = stem.lower()
    if all(t in low for t in ["adaptive", "pag"]):
        return 4
    _tokens = {1: ["naive"], 2: ["default"], 3: ["pag", "enhanced"]}
    for mode_id, tokens in _tokens.items():
        if all(t in low for t in tokens):
            return mode_id
    return None


def _pag_scale_from_sidecar(meta: dict[str, str]) -> Optional[float]:
    m = re.search(r"scale\s*=\s*([\d.]+)", meta.get("Attention", ""), re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _parse_inference_time(meta: dict[str, str]) -> Optional[float]:
    raw = meta.get("Time", "")
    try:
        return float(raw.replace("s", "").strip())
    except ValueError:
        return None


def _parse_prompt(meta: dict[str, str]) -> Optional[str]:
    return meta.get("Full Prompt") or None


def _parse_sharpness(meta: dict[str, str]) -> Optional[float]:
    """Use pre-computed sharpness from sidecar if available."""
    try:
        return float(meta.get("Sharpness", ""))
    except ValueError:
        return None


# =============================================================================
# FID helpers
# =============================================================================

class _ImageFolderDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


def _build_inception(device):
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    m = models.inception_v3(weights=weights, transform_input=False, aux_logits=True)
    m.fc = torch.nn.Identity()
    return m.eval().to(device)


def _inception_transform():
    return transforms.Compose([
        transforms.Resize(FID_IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(FID_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.inference_mode()
def _extract_features(paths, model, device):
    loader = DataLoader(_ImageFolderDataset(paths, _inception_transform()),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    feats = []
    for batch in loader:
        out = model(batch.to(device))
        feat = out if isinstance(out, torch.Tensor) else (out.logits if hasattr(out, "logits") else out[0])
        feats.append(feat.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def _sqrtm_psd(mat):
    mat = (mat + mat.T) * 0.5
    eigvals, eigvecs = linalg.eigh(mat)
    return eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0, None))) @ eigvecs.T


def compute_fid(real_paths, gen_paths, device, inception_model) -> float:
    if len(real_paths) < 2 or len(gen_paths) < 2:
        return float("nan")
    rf = _extract_features(real_paths, inception_model, device)
    gf = _extract_features(gen_paths, inception_model, device)
    mu1, mu2 = rf.mean(0), gf.mean(0)
    s1, s2   = np.cov(rf, rowvar=False), np.cov(gf, rowvar=False)
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(s1 + s2 - 2.0 * _sqrtm_psd(s1 @ s2)))


# =============================================================================
# SSIM / CLIP helpers
# =============================================================================

def _compute_ssim_best_match(real_paths, gen_path) -> float:
    size = (SSIM_IMAGE_SIZE, SSIM_IMAGE_SIZE)
    gen_arr = np.asarray(Image.open(gen_path).convert("RGB").resize(size, Image.Resampling.LANCZOS))
    best = -1.0
    for rp in real_paths:
        r_arr = np.asarray(Image.open(rp).convert("RGB").resize(size, Image.Resampling.LANCZOS))
        try:
            s = float(ssim(r_arr, gen_arr, channel_axis=2, data_range=255))
        except TypeError:
            s = float(ssim(r_arr, gen_arr, multichannel=True, data_range=255))
        if s > best:
            best = s
    return best


@torch.inference_mode()
def _compute_clip_scores(paths, prompts, clip_model, clip_processor, device) -> list[float]:
    scores = []
    for path, prompt in zip(paths, prompts):
        if not prompt:
            continue
        image  = Image.open(path).convert("RGB")
        inputs = clip_processor(text=[prompt], images=image, return_tensors="pt",
                                padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        img_f  = F.normalize(clip_model.get_image_features(pixel_values=inputs["pixel_values"]), dim=-1)
        txt_f  = F.normalize(clip_model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")), dim=-1)
        scores.append(float((img_f * txt_f).sum(dim=-1).item()))
    return scores


# =============================================================================
# Data collection
# =============================================================================

@dataclass
class ModeSamples:
    mode_id: int
    paths:        list[Path]           = field(default_factory=list)
    prompts:      list[Optional[str]]  = field(default_factory=list)
    infer_times:  list[Optional[float]]= field(default_factory=list)
    sharpness_pre: list[Optional[float]]= field(default_factory=list)  # from sidecar


def collect_samples(gen_dir, prefix, pag_scale) -> dict[int, ModeSamples]:
    if not gen_dir.is_dir():
        raise FileNotFoundError(f"Generated folder not found: {gen_dir}")
    buckets = {m: ModeSamples(mode_id=m) for m in range(1, 5)}
    for p in sorted(gen_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not p.name.startswith(prefix):
            continue
        meta = _read_sidecar(p)
        mode = _mode_from_sidecar(meta) or _mode_from_filename(p.stem)
        if mode is None or mode not in range(1, 5):
            continue
        # PAG Enhanced: only the requested scale
        if mode == 3:
            scale = _pag_scale_from_sidecar(meta)
            if scale is not None and abs(scale - pag_scale) > 1e-3:
                continue
        bucket = buckets[mode]
        if len(bucket.paths) >= MAX_SAMPLES_PER_MODE:
            continue
        bucket.paths.append(p)
        bucket.prompts.append(_parse_prompt(meta))
        bucket.infer_times.append(_parse_inference_time(meta))
        bucket.sharpness_pre.append(_parse_sharpness(meta))
    return buckets


# =============================================================================
# Per-mode statistics
# =============================================================================

@dataclass
class ModeStats:
    mode_id: int
    n: int
    sharpness_mean: float; sharpness_std: float
    ssim_mean: float;      ssim_std:  float
    clip_mean: float;      clip_std:  float
    infer_mean: Optional[float]; infer_std: Optional[float]


def compute_mode_stats(bucket, real_paths, clip_model, clip_processor, device) -> ModeStats:
    n = len(bucket.paths)
    nan = float("nan")
    if n == 0:
        return ModeStats(bucket.mode_id, 0, nan, nan, nan, nan, nan, nan, None, None)

    # Sharpness: prefer sidecar value, else compute from image
    sharpness_vals = []
    for i, p in enumerate(bucket.paths):
        pre = bucket.sharpness_pre[i]
        if pre is not None:
            sharpness_vals.append(pre)
        else:
            from scipy.ndimage import laplace
            arr = np.asarray(Image.open(p).convert("RGB"))
            gray = (0.2126*arr[:,:,0] + 0.7152*arr[:,:,1] + 0.0722*arr[:,:,2]).astype(np.float32)
            sharpness_vals.append(float(laplace(gray).var()))

    ssim_vals = [_compute_ssim_best_match(real_paths, p) for p in bucket.paths]

    clip_inputs = [(p, pr) for p, pr in zip(bucket.paths, bucket.prompts) if pr]
    clip_vals   = _compute_clip_scores(
        [x[0] for x in clip_inputs], [x[1] for x in clip_inputs],
        clip_model, clip_processor, device
    ) if clip_inputs else []

    infer_vals = [t for t in bucket.infer_times if t is not None]
    return ModeStats(
        mode_id=bucket.mode_id, n=n,
        sharpness_mean=float(np.mean(sharpness_vals)),
        sharpness_std =float(np.std(sharpness_vals, ddof=1)) if n > 1 else 0.0,
        ssim_mean=float(np.mean(ssim_vals)),
        ssim_std =float(np.std(ssim_vals, ddof=1)) if n > 1 else 0.0,
        clip_mean=float(np.mean(clip_vals)) if clip_vals else nan,
        clip_std =float(np.std(clip_vals, ddof=1)) if len(clip_vals) > 1 else nan,
        infer_mean=float(np.mean(infer_vals)) if infer_vals else None,
        infer_std =float(np.std(infer_vals, ddof=1)) if len(infer_vals) > 1 else None,
    )


# =============================================================================
# Output
# =============================================================================

def _fmt(mean, std, d=4):
    if np.isnan(mean): return "N/A"
    if np.isnan(std):  return f"{mean:.{d}f}"
    return f"{mean:.{d}f} ± {std:.{d}f}"

def _fmt_time(mean, std):
    if mean is None: return "N/A"
    if std  is None: return f"{mean:.2f}s"
    return f"{mean:.2f}s ± {std:.2f}s"


def print_table(stats_list):
    W = 44
    sep = "-" * (W + 4*24)
    hdr = f"{'Mode':<{W}} {'Sharpness ↑':>24} {'SSIM ↑':>24} {'CLIP ↑':>24} {'Time ↓':>24}"
    print(f"\n{'#'*len(hdr)}")
    print(f"Table 4.2 — Attention Mode Quantitative Benchmark ({ITEM})")
    print(f"PAG Enhanced uses scale={PAG_SCALE_FOR_COMPARISON} | mean ± std | up to {MAX_SAMPLES_PER_MODE} samples/mode")
    print('#'*len(hdr))
    print(hdr)
    print(sep)
    for s in stats_list:
        print(
            f"{MODE_NAMES[s.mode_id]:<{W}}"
            f" {_fmt(s.sharpness_mean, s.sharpness_std, 2):>24}"
            f" {_fmt(s.ssim_mean, s.ssim_std):>24}"
            f" {_fmt(s.clip_mean, s.clip_std):>24}"
            f" {_fmt_time(s.infer_mean, s.infer_std):>24}"
            f"  (n={s.n})"
        )
    print(sep)
    print("Sharpness=Laplacian var | SSIM=best-match vs real refs | CLIP=image–text cosine")


def save_csv(stats_list):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Mode","N","Sharpness Mean","Sharpness Std",
                    "SSIM Mean","SSIM Std","CLIP Mean","CLIP Std",
                    "Time Mean (s)","Time Std (s)"])
        for s in stats_list:
            def fv(v): return f"{v:.4f}" if v is not None and not np.isnan(v) else "N/A"
            w.writerow([MODE_NAMES[s.mode_id], s.n,
                        fv(s.sharpness_mean), fv(s.sharpness_std),
                        fv(s.ssim_mean),      fv(s.ssim_std),
                        fv(s.clip_mean),      fv(s.clip_std),
                        fv(s.infer_mean),     fv(s.infer_std)])
    print(f"\n✓ CSV saved → {OUTPUT_CSV}")


# =============================================================================
# Main
# =============================================================================

def main():
    device = torch.device(DEVICE)
    print(f"Item         : {ITEM}")
    print(f"PAG scale    : {PAG_SCALE_FOR_COMPARISON}  (for PAG Enhanced cross-mode comparison)")
    print(f"Real dir     : {REAL_DIR}")
    print(f"Gen dir      : {GEN_DIR}")
    print(f"Device       : {device}")

    if not REAL_DIR.is_dir():
        raise FileNotFoundError(f"Real image directory not found: {REAL_DIR}")
    real_paths = [p for p in sorted(REAL_DIR.iterdir())
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not real_paths:
        raise ValueError(f"No reference images found in {REAL_DIR}")
    print(f"Real images  : {len(real_paths)}")

    print("\nLoading CLIP model...")
    clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_ID).eval().to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    buckets = collect_samples(GEN_DIR, GEN_PREFIX, PAG_SCALE_FOR_COMPARISON)
    for m, b in sorted(buckets.items()):
        print(f"  Mode {m}: {len(b.paths)} images")

    stats_list = []
    for mode_id in range(1, 5):
        b = buckets[mode_id]
        if not b.paths:
            print(f"\n⚠  No images for Mode {mode_id} — skipping.")
            nan = float("nan")
            stats_list.append(ModeStats(mode_id, 0, nan, nan, nan, nan, nan, nan, None, None))
            continue
        print(f"\nMode {mode_id} ({len(b.paths)} images)...")
        t0 = time.perf_counter()
        s  = compute_mode_stats(b, real_paths, clip_model, clip_processor, device)
        print(f"  Done {time.perf_counter()-t0:.1f}s — sharpness={s.sharpness_mean:.2f}, "
              f"SSIM={s.ssim_mean:.4f}, CLIP={s.clip_mean:.4f}")
        stats_list.append(s)

    print_table(stats_list)
    if SAVE_CSV:
        save_csv(stats_list)


if __name__ == "__main__":
    main()
