"""
Evaluate generated images with FID, SSIM, and CLIP Score.

Only change PRESET (1–5) below, then run:  python evaluate_metrics.py

Each run automatically compares two groups from output/kd_gen:
  - With LoRA    (sidecar txt line: LoRA: Yes ...)
  - Without LoRA (sidecar txt line: LoRA: No ...)
"""

from __future__ import annotations

from dataclasses import dataclass
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
# ★ ONLY CHANGE THIS LINE ★
# =============================================================================
PRESET = 1
# 1 = Sword    → data/sword/     output/kd_gen/Sword_*.png
# 2 = Bow      → data/bow/       output/kd_gen/Bow_*.png
# 3 = Polearm  → data/polearm/   output/kd_gen/Polearm_*.png
# 4 = Armor    → data/armor/     output/kd_gen/Armor_*.png
# 5 = Shield   → data/shield/    output/kd_gen/Shield_*.png

# Optional: only count generated files whose filename contains this text.
# Examples: "Ancient", "PAG_Enhanced", "Default"
# Set to None to include every generation for the selected item.
GEN_NAME_FILTER = None

PROJECT_ROOT = Path(__file__).resolve().parent

if PRESET == 1:
    ITEM = "Sword"
    REAL_DIR = PROJECT_ROOT / "data" / "sword"
elif PRESET == 2:
    ITEM = "Bow"
    REAL_DIR = PROJECT_ROOT / "data" / "bow"
elif PRESET == 3:
    ITEM = "Polearm"
    REAL_DIR = PROJECT_ROOT / "data" / "polearm"
elif PRESET == 4:
    ITEM = "Armor"
    REAL_DIR = PROJECT_ROOT / "data" / "armor"
elif PRESET == 5:
    ITEM = "Shield"
    REAL_DIR = PROJECT_ROOT / "data" / "shield"
else:
    raise ValueError("PRESET must be 1–5 (Sword / Bow / Polearm / Armor / Shield)")

# Generated images saved by app.py — do not change unless you changed save path in app.py
GEN_DIR = PROJECT_ROOT / "output" / "kd_gen"
GEN_PREFIX = f"{ITEM}_"

# Model / runtime
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FID_IMAGE_SIZE = 299
SSIM_IMAGE_SIZE = 512

# Max real reference images sampled for SSIM best-match.
# 166 real × 137 gen = 22 k comparisons at 512px — very slow.
# 30 references gives statistically stable results in ~2 min.
MAX_REAL_FOR_SSIM = 30

# =============================================================================
# Helpers
# =============================================================================


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    files = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not files:
        raise ValueError(f"No images found in {folder}")
    return files


def list_generated_images(
    folder: Path,
    prefix: str,
    name_filter: Optional[str] = None,
    lora_used: Optional[bool] = None,
) -> list[Path]:
    """Pick png/jpg from kd_gen; optionally filter by filename and LoRA sidecar."""
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Generated folder not found: {folder}\n"
            "Run app.py and generate some images first."
        )
    files: list[Path] = []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not p.name.startswith(prefix):
            continue
        if name_filter and name_filter not in p.name:
            continue
        if lora_used is not None:
            sidecar_lora = parse_lora_from_sidecar(p.with_suffix(".txt"))
            if sidecar_lora != lora_used:
                continue
        files.append(p)
    if not files:
        lora_hint = ""
        if lora_used is True:
            lora_hint = ", LoRA: Yes"
        elif lora_used is False:
            lora_hint = ", LoRA: No"
        name_hint = f" matching '{name_filter}'" if name_filter else ""
        raise ValueError(
            f"No generated images found in {folder}\n"
            f"  looking for: {prefix}*.png{name_hint}{lora_hint}\n"
            "Generate both With/Without LoRA in app.py first."
        )
    return files


def parse_lora_from_sidecar(sidecar: Path) -> Optional[bool]:
    """Return True if LoRA was used, False if disabled, None if sidecar missing."""
    if not sidecar.is_file():
        return None
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        if line.startswith("LoRA:"):
            if "Yes" in line:
                return True
            if "No" in line:
                return False
    return None


def parse_prompt_from_sidecar(sidecar: Path) -> Optional[str]:
    """Read 'Full Prompt: ...' line written by app.py next to each generated png."""
    if not sidecar.is_file():
        return None
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        if line.startswith("Full Prompt:"):
            return line.split("Full Prompt:", 1)[1].strip()
    return None


def load_rgb(path: Path, size: Optional[tuple[int, int]] = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(img)


@dataclass
class MetricResults:
    label: str
    image_count: int
    fid: Optional[float]
    ssim_mean: Optional[float]
    ssim_pairs: Optional[int]
    clip_score_mean: Optional[float]
    clip_pairs: Optional[int]


# =============================================================================
# FID (Fréchet Inception Distance)
# Reference: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule..."
# =============================================================================


class _ImageFolderDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def _build_inception(device: torch.device) -> torch.nn.Module:
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    # Pretrained IMAGENET1K_V1 weights require aux_logits=True
    model = models.inception_v3(weights=weights, transform_input=False, aux_logits=True)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


def _inception_transform() -> transforms.Compose:
    # Match standard FID preprocessing ([-1, 1] after normalize)
    return transforms.Compose([
        transforms.Resize(FID_IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(FID_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.inference_mode()
def _extract_inception_features(
    paths: list[Path],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        _ImageFolderDataset(paths, _inception_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    feats: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        # Inception expects 299x299; forward returns logits during training only
        out = model(batch)
        if isinstance(out, torch.Tensor):
            feat = out
        else:
            feat = out.logits if hasattr(out, "logits") else out[0]
        feats.append(feat.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def _covariance(features: np.ndarray) -> np.ndarray:
    if features.shape[0] < 2:
        raise ValueError("FID needs at least 2 images per set to estimate covariance.")
    return np.cov(features, rowvar=False)


def _sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
    """Matrix square root for positive semi-definite matrices."""
    matrix = (matrix + matrix.T) * 0.5
    eigvals, eigvecs = linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 0, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def compute_fid(
    real_paths: list[Path],
    gen_paths: list[Path],
    device: torch.device,
    inception_model: Optional[torch.nn.Module] = None,
) -> float:
    model = inception_model or _build_inception(device)
    real_feat = _extract_inception_features(real_paths, model, device, BATCH_SIZE)
    gen_feat = _extract_inception_features(gen_paths, model, device, BATCH_SIZE)

    mu1, mu2 = real_feat.mean(axis=0), gen_feat.mean(axis=0)
    sigma1, sigma2 = _covariance(real_feat), _covariance(gen_feat)

    diff = mu1 - mu2
    covmean = _sqrtm_psd(sigma1 @ sigma2)
    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return fid


# =============================================================================
# SSIM (Structural Similarity Index)
# Reference: Wang et al., "Image Quality Assessment: From Error Visibility..."
# =============================================================================


def _ssim_pair_arrays(real_arr: np.ndarray, gen_arr: np.ndarray) -> float:
    """SSIM between two pre-loaded, pre-resized uint8 arrays."""
    try:
        return float(ssim(real_arr, gen_arr, channel_axis=2, data_range=255))
    except TypeError:
        return float(ssim(real_arr, gen_arr, multichannel=True, data_range=255))


def compute_ssim_best_match(real_paths: list[Path], gen_paths: list[Path]) -> tuple[float, int]:
    """
    For each generated image, find the highest SSIM among sampled real references,
    then average those best-match scores.

    Speed-ups vs original:
      1. Subsample real refs to MAX_REAL_FOR_SSIM (random, seeded for reproducibility).
      2. Pre-load and pre-resize ALL real arrays once, before the gen loop.
      3. Pre-resize each gen image once per gen image, not once per (real, gen) pair.
    """
    import random

    size = (SSIM_IMAGE_SIZE, SSIM_IMAGE_SIZE)

    # Subsample real references
    sampled_reals = real_paths
    if len(real_paths) > MAX_REAL_FOR_SSIM:
        rng = random.Random(42)
        sampled_reals = rng.sample(real_paths, MAX_REAL_FOR_SSIM)
        print(f"  [SSIM] subsampled {MAX_REAL_FOR_SSIM}/{len(real_paths)} real refs for speed")

    # Pre-load all real arrays once
    print(f"  [SSIM] pre-loading {len(sampled_reals)} real reference images...")
    real_arrays = [
        np.asarray(Image.open(p).convert("RGB").resize(size, Image.Resampling.LANCZOS))
        for p in sampled_reals
    ]

    scores: list[float] = []
    total = len(gen_paths)
    for i, gen_p in enumerate(gen_paths, 1):
        gen_arr = np.asarray(Image.open(gen_p).convert("RGB").resize(size, Image.Resampling.LANCZOS))
        best = max(_ssim_pair_arrays(r, gen_arr) for r in real_arrays)
        scores.append(best)
        if i % 10 == 0 or i == total:
            print(f"  [SSIM] {i}/{total} done", end="\r")

    print()  # newline after progress
    return float(np.mean(scores)), len(scores)


# =============================================================================
# CLIP Score (image–text cosine similarity)
# Reference: Hessel et al., "CLIPScore: A Reference-free Evaluation Metric..."
# =============================================================================


@torch.inference_mode()
def compute_clip_score(
    gen_paths: list[Path],
    model_id: str,
    device: torch.device,
    clip_bundle: Optional[tuple[CLIPModel, CLIPProcessor]] = None,
) -> tuple[float, int]:
    if clip_bundle is None:
        model = CLIPModel.from_pretrained(model_id).eval().to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
    else:
        model, processor = clip_bundle

    scores: list[float] = []
    for path in gen_paths:
        prompt = parse_prompt_from_sidecar(path.with_suffix(".txt"))
        if not prompt:
            print(f"  [skip CLIP] no 'Full Prompt:' in {path.with_suffix('.txt').name}")
            continue
        image = Image.open(path).convert("RGB")
        inputs = processor(text=[prompt], images=image, return_tensors="pt",
                           padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        score = (image_features * text_features).sum(dim=-1).item()
        scores.append(score)

    if not scores:
        return float("nan"), 0
    return float(np.mean(scores)), len(scores)


# =============================================================================
# Main
# =============================================================================


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.{digits}f}"


def evaluate_group(
    label: str,
    lora_used: bool,
    real_paths: list[Path],
    device: torch.device,
    inception_model: torch.nn.Module,
    clip_bundle: tuple[CLIPModel, CLIPProcessor],
) -> MetricResults:
    gen_paths = list_generated_images(
        GEN_DIR, GEN_PREFIX, GEN_NAME_FILTER, lora_used=lora_used
    )

    print(f"\n{'=' * 50}")
    print(f"{label}  ({len(gen_paths)} images)")
    print("=" * 50)

    fid_value: Optional[float] = None
    if len(real_paths) >= 2 and len(gen_paths) >= 2:
        print("Computing FID...")
        fid_value = compute_fid(real_paths, gen_paths, device, inception_model)
        print(f"  FID = {fid_value:.4f}  (lower is better)")
    else:
        print("Skipping FID (need >= 2 images in each folder).")

    print("Computing SSIM (best match vs all real references)...")
    ssim_mean, ssim_n = compute_ssim_best_match(real_paths, gen_paths)
    print(f"  SSIM = {ssim_mean:.4f}  over {ssim_n} generated images  (higher is better, max 1.0)")

    print("Computing CLIP Score (prompt from each .txt sidecar)...")
    clip_mean, clip_n = compute_clip_score(
        gen_paths, CLIP_MODEL_ID, device, clip_bundle=clip_bundle
    )
    print(f"  CLIP Score = {clip_mean:.4f}  over {clip_n} images  (higher is better)")

    return MetricResults(
        label=label,
        image_count=len(gen_paths),
        fid=fid_value,
        ssim_mean=ssim_mean if ssim_n else None,
        ssim_pairs=ssim_n or None,
        clip_score_mean=clip_mean if clip_n else None,
        clip_pairs=clip_n or None,
    )


def print_comparison(
    item: str,
    with_lora: Optional[MetricResults],
    without_lora: Optional[MetricResults],
) -> None:
    print(f"\n{'#' * 50}")
    print(f"LoRA COMPARISON — {item}")
    print("#" * 50)
    print(f"{'Metric':<14} {'With LoRA':>16} {'Without LoRA':>16} {'Better':>10}")
    print("-" * 58)

    rows = [
        ("Images", "image_count", None),
        ("FID ↓", "fid", "lower"),
        ("SSIM ↑", "ssim_mean", "higher"),
        ("CLIP ↑", "clip_score_mean", "higher"),
    ]
    for name, field, direction in rows:
        if field == "image_count":
            a = str(with_lora.image_count) if with_lora else "N/A"
            b = str(without_lora.image_count) if without_lora else "N/A"
            better = "-"
        else:
            va = getattr(with_lora, field, None) if with_lora else None
            vb = getattr(without_lora, field, None) if without_lora else None
            a, b = _fmt(va), _fmt(vb)
            better = "-"
            if va is not None and vb is not None and not (np.isnan(va) or np.isnan(vb)):
                if direction == "lower":
                    better = "With LoRA" if va < vb else "Without LoRA"
                    if va == vb:
                        better = "Tie"
                elif direction == "higher":
                    better = "With LoRA" if va > vb else "Without LoRA"
                    if va == vb:
                        better = "Tie"
        print(f"{name:<14} {a:>16} {b:>16} {better:>10}")

    print("-" * 58)
    print("FID: lower is better | SSIM / CLIP: higher is better")


def evaluate() -> tuple[Optional[MetricResults], Optional[MetricResults]]:
    device = torch.device(DEVICE)
    real_paths = list_images(REAL_DIR)

    print(f"PRESET      : {PRESET} → {ITEM}")
    print(f"原图 REAL   : {len(real_paths)} images in {REAL_DIR}")
    print(f"生成图 GEN  : {GEN_DIR}  (prefix {GEN_PREFIX}*)")
    if GEN_NAME_FILTER:
        print(f"GEN filter  : filename contains '{GEN_NAME_FILTER}'")
    print(f"Device      : {device}")

    print("\nLoading models (shared for both LoRA groups)...")
    inception_model = _build_inception(device)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).eval().to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    clip_bundle = (clip_model, clip_processor)

    with_lora: Optional[MetricResults] = None
    without_lora: Optional[MetricResults] = None

    try:
        with_lora = evaluate_group(
            "With LoRA", True, real_paths, device, inception_model, clip_bundle
        )
    except ValueError as exc:
        print(f"\n[skip With LoRA] {exc}")

    try:
        without_lora = evaluate_group(
            "Without LoRA", False, real_paths, device, inception_model, clip_bundle
        )
    except ValueError as exc:
        print(f"\n[skip Without LoRA] {exc}")

    if with_lora is None and without_lora is None:
        raise ValueError("No images found for either LoRA group.")

    print_comparison(ITEM, with_lora, without_lora)
    return with_lora, without_lora


if __name__ == "__main__":
    evaluate()