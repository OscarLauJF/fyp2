"""
Table 4.3 — LoRA Strength Ablation
=====================================
FID and CLIP Score for LoRA strengths ∈ {0.5, 0.6, 0.7, 0.8, 0.9}
showing image quality vs strength tradeoff.

Sidecar .txt format (written by app.py):
    LoRA: Yes (armor-000106) strength=0.80
    Attention: <mode>
    Full Prompt: <text>
    Time: <float>s

Filename pattern:
    {Item}_{Style}_{Mode tokens}_{seed1}_{seed2}.png

All LoRA-enabled images for the selected item are bucketed by strength.
Non-LoRA images (LoRA: No) are ignored.

Change ITEM_PRESET, then run:  python table4_3_lora_ablation.py
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import CLIPModel, CLIPProcessor

# =============================================================================
# ★ ONLY CHANGE THESE LINES ★
# =============================================================================
ITEM_PRESET = 1
# 1 = Sword   2 = Bow   3 = Polearm   4 = Armor   5 = Shield

LORA_STRENGTHS = [0.5, 0.6, 0.7, 0.8, 0.9]
SAVE_CSV       = True
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

OUTPUT_DIR = PROJECT_ROOT / "output" / "tables"
OUTPUT_CSV = OUTPUT_DIR / "table4_3_lora_ablation.csv"


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


def _parse_lora_enabled(meta: dict[str, str]) -> Optional[bool]:
    """
    Parse 'LoRA: Yes (armor-000106) strength=0.80' or 'LoRA: No (disabled)'.
    Returns True/False/None.
    """
    raw = meta.get("LoRA", "").lower()
    if not raw:
        return None
    if raw.startswith("yes"):
        return True
    if raw.startswith("no"):
        return False
    return None


def _parse_lora_strength(meta: dict[str, str]) -> Optional[float]:
    """
    Extract strength from 'LoRA: Yes (armor-000106) strength=0.80'.
    """
    raw = meta.get("LoRA", "")
    m = re.search(r"strength\s*=\s*([\d.]+)", raw, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _parse_prompt(meta: dict[str, str]) -> Optional[str]:
    return meta.get("Full Prompt") or None


# =============================================================================
# FID helpers  (same pattern as evaluate_metrics.py)
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
        out  = model(batch.to(device))
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
    gf = _extract_features(gen_paths,  inception_model, device)
    mu1, mu2 = rf.mean(0), gf.mean(0)
    s1, s2   = np.cov(rf, rowvar=False), np.cov(gf, rowvar=False)
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(s1 + s2 - 2.0 * _sqrtm_psd(s1 @ s2)))


# =============================================================================
# CLIP score
# =============================================================================

@torch.inference_mode()
def compute_clip_score(gen_paths, prompts, model, processor, device):
    scores = []
    for path, prompt in zip(gen_paths, prompts):
        if not prompt:
            continue
        image  = Image.open(path).convert("RGB")
        inputs = processor(text=[prompt], images=image, return_tensors="pt",
                           padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        img_f  = F.normalize(model.get_image_features(pixel_values=inputs["pixel_values"]), dim=-1)
        txt_f  = F.normalize(model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")), dim=-1)
        scores.append(float((img_f * txt_f).sum(dim=-1).item()))
    nan = float("nan")
    if not scores:
        return nan, nan
    return float(np.mean(scores)), float(np.std(scores, ddof=1)) if len(scores) > 1 else nan


# =============================================================================
# Data collection
# =============================================================================

@dataclass
class StrengthBucket:
    strength: float
    paths:    list[Path]          = field(default_factory=list)
    prompts:  list[Optional[str]] = field(default_factory=list)


def collect_by_strength(gen_dir, prefix, strengths) -> dict[float, StrengthBucket]:
    if not gen_dir.is_dir():
        raise FileNotFoundError(f"Generated folder not found: {gen_dir}")

    buckets: dict[float, StrengthBucket] = {s: StrengthBucket(strength=s) for s in strengths}
    unrecognised = 0

    for p in sorted(gen_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not p.name.startswith(prefix):
            continue
        meta    = _read_sidecar(p)
        enabled = _parse_lora_enabled(meta)
        if enabled is not True:
            continue   # skip LoRA-disabled images
        strength = _parse_lora_strength(meta)
        if strength is None:
            unrecognised += 1
            continue
        # Round to 1 decimal place to avoid float precision issues (0.7000000001 etc.)
        strength = round(strength, 1)
        if strength not in buckets:
            unrecognised += 1
            continue
        buckets[strength].paths.append(p)
        buckets[strength].prompts.append(_parse_prompt(meta))

    if unrecognised:
        print(f"  [info] {unrecognised} LoRA-enabled images had unrecognised/unlisted strength — skipped.")
    return buckets


# =============================================================================
# Result + output
# =============================================================================

@dataclass
class StrengthStats:
    strength: float
    n: int
    fid: float
    clip_mean: float
    clip_std: float


def _fv(v, d=4):
    return "N/A" if v is None or np.isnan(v) else f"{v:.{d}f}"

def _fvs(m, s, d=4):
    if np.isnan(m): return "N/A"
    if np.isnan(s): return f"{m:.{d}f}"
    return f"{m:.{d}f} ± {s:.{d}f}"


def print_table(stats):
    sep = "-" * 62
    hdr = f"{'Strength':<12} {'N':>5} {'FID ↓':>14} {'CLIP Score ↑':>22}"
    print(f"\n{'#'*62}")
    print(f"Table 4.3 — LoRA Strength Ablation ({ITEM})")
    print(f"Image quality metrics vs LoRA strength")
    print('#'*62)
    print(hdr)
    print(sep)
    for s in stats:
        print(f"{s.strength:<12.1f} {s.n:>5} {_fv(s.fid, 2):>14} "
              f"{_fvs(s.clip_mean, s.clip_std):>22}")
    print(sep)
    print("FID ↓ lower is better | CLIP ↑ higher is better")


def save_csv(stats):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["LoRA Strength", "N", "FID", "CLIP Mean", "CLIP Std"])
        for s in stats:
            w.writerow([s.strength, s.n,
                        _fv(s.fid), _fv(s.clip_mean), _fv(s.clip_std)])
    print(f"\n✓ CSV saved → {OUTPUT_CSV}")


# =============================================================================
# Main
# =============================================================================

def main():
    device = torch.device(DEVICE)
    print(f"Item         : {ITEM}")
    print(f"LoRA strengths: {LORA_STRENGTHS}")
    print(f"Real dir     : {REAL_DIR}")
    print(f"Gen dir      : {GEN_DIR}")
    print(f"Device       : {device}")

    if not REAL_DIR.is_dir():
        raise FileNotFoundError(f"Real image directory not found: {REAL_DIR}")
    real_paths = [p for p in sorted(REAL_DIR.iterdir())
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not real_paths:
        raise ValueError(f"No reference images found in {REAL_DIR}")
    print(f"Real imgs    : {len(real_paths)}")

    print("\nLoading models...")
    inception_model = _build_inception(device)
    clip_model      = CLIPModel.from_pretrained(CLIP_MODEL_ID).eval().to(device)
    clip_processor  = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    buckets = collect_by_strength(GEN_DIR, GEN_PREFIX, LORA_STRENGTHS)
    for s, b in sorted(buckets.items()):
        print(f"  Strength {s}: {len(b.paths)} images")

    stats = []
    nan   = float("nan")
    for strength in sorted(LORA_STRENGTHS):
        b = buckets[strength]
        if not b.paths:
            print(f"\n⚠  No images for strength {strength} — skipping.")
            stats.append(StrengthStats(strength, 0, nan, nan, nan))
            continue
        print(f"\nStrength {strength} ({len(b.paths)} images)...")
        fid = compute_fid(real_paths, b.paths, device, inception_model)
        print(f"  FID  = {fid:.4f}" if not np.isnan(fid) else "  FID  = N/A")
        cm, cs = compute_clip_score(b.paths, b.prompts, clip_model, clip_processor, device)
        print(f"  CLIP = {cm:.4f} ± {cs:.4f}" if not np.isnan(cm) else "  CLIP = N/A")
        stats.append(StrengthStats(strength, len(b.paths), fid, cm, cs))

    print_table(stats)
    if SAVE_CSV:
        save_csv(stats)


if __name__ == "__main__":
    main()