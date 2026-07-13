"""
Figure 4.2 — Qualitative Comparison Across Four Attention Modes
================================================================
Loads one generated image per attention mode (same prompt, same seed)
and assembles a 2×2 side-by-side panel saved as:
    output/figures/figure4_2_attention_mode_comparison.png

Sidecar .txt format (written by app.py):
    Attention: <Naive (Baseline) | Default (AttnProcessor2_0) |
                PAG Enhanced (scale=X.X) | Adaptive PAG + Progressive CFG (Proposed method)>
    Seed: <int>
    Full Prompt: <text>
    ...

Filename pattern:
    {Item}_{Style}_{Mode tokens}_{seed1}_{seed2}.png
    e.g. Sword_Ancient_PAG_Enhanced_1854335366_1781094831.png

For PAG Enhanced images, only scale=2.5 is used in this figure.
Change ITEM_PRESET to switch weapon type, then run:
    python figure4_2_attention_mode_comparison.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# =============================================================================
# ★ ONLY CHANGE THESE LINES ★
# =============================================================================
ITEM_PRESET = 1
# 1 = Sword   2 = Bow   3 = Polearm   4 = Armor   5 = Shield

# Prefer images matching this seed (int).  None = first found per mode.
TARGET_SEED: Optional[int] = None

# PAG scale to use when selecting PAG Enhanced images for this figure.
PAG_SCALE_FOR_COMPARISON = 2.5
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

_ITEM_MAP = {1: "Sword", 2: "Bow", 3: "Polearm", 4: "Armor", 5: "Shield"}
if ITEM_PRESET not in _ITEM_MAP:
    raise ValueError("ITEM_PRESET must be 1–5")

ITEM       = _ITEM_MAP[ITEM_PRESET]
GEN_DIR    = PROJECT_ROOT / "output" / "kd_gen"
GEN_PREFIX = f"{ITEM}_"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
OUTPUT_DIR  = PROJECT_ROOT / "output" / "figures"
OUTPUT_PATH = OUTPUT_DIR / "figure4_2_attention_mode_comparison.png"

MODE_LABELS = {
    1: "Mode 1 — Naive",
    2: "Mode 2 — Default",
    3: "Mode 3 — PAG Enhanced",
    4: "Mode 4 — Adaptive PAG\n+ Progressive CFG",
}
BORDER_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

# ── Filename keyword sets per mode ──────────────────────────────────────────
# Each set: ALL tokens must appear (case-insensitive) in the stem.
_MODE_FILENAME_TOKENS: dict[int, list[str]] = {
    1: ["naive"],
    2: ["default"],
    3: ["pag", "enhanced"],
    4: ["adaptive", "pag"],   # also matches "Adaptive_PAG_+_Progressive_CFG"
}

# ── Sidecar "Attention:" values per mode ────────────────────────────────────
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
    """Parse every  Key: value  line from the companion .txt sidecar."""
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
    """
    Parse attention mode from the sidecar 'Attention:' line.
    Examples:
        Attention: Naive (Baseline)
        Attention: Default (AttnProcessor2_0)
        Attention: PAG Enhanced (scale=3.0)
        Attention: Adaptive PAG + Progressive CFG (Proposed method)
    """
    raw = meta.get("Attention", "").lower()
    if not raw:
        return None
    for mode_id, keywords in _MODE_ATTENTION_STRINGS.items():
        if all(kw in raw for kw in keywords):
            # For mode 4, make sure it's not accidentally matching mode 3
            if mode_id == 3 and "adaptive" in raw:
                continue
            return mode_id
    return None


def _mode_from_filename(stem: str) -> Optional[int]:
    """Fallback: infer mode from filename tokens."""
    low = stem.lower()
    # Check mode 4 first (its tokens are a superset of mode 3's "pag")
    if all(t in low for t in ["adaptive", "pag"]):
        return 4
    for mode_id, tokens in _MODE_FILENAME_TOKENS.items():
        if mode_id == 4:
            continue  # already handled
        if all(t in low for t in tokens):
            return mode_id
    return None


def _pag_scale_from_sidecar(meta: dict[str, str]) -> Optional[float]:
    """Extract scale from 'Attention: PAG Enhanced (scale=X.X)'."""
    raw = meta.get("Attention", "")
    import re
    m = re.search(r"scale\s*=\s*([\d.]+)", raw, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _seed_from_sidecar(meta: dict[str, str]) -> Optional[int]:
    try:
        return int(meta.get("Seed", ""))
    except ValueError:
        return None


# =============================================================================
# Image selection
# =============================================================================

def collect_mode_images(
    gen_dir: Path,
    prefix: str,
    target_seed: Optional[int],
    pag_scale_for_comparison: float,
) -> dict[int, Path]:
    """
    Return {mode_int: image_path} for modes 1–4.
    PAG Enhanced (mode 3) is filtered to pag_scale_for_comparison.
    """
    if not gen_dir.is_dir():
        raise FileNotFoundError(
            f"Generated folder not found: {gen_dir}\n"
            "Run app.py and generate images for all four attention modes first."
        )

    mode_to_path: dict[int, Path] = {}

    for p in sorted(gen_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not p.name.startswith(prefix):
            continue

        meta  = _read_sidecar(p)
        mode  = _mode_from_sidecar(meta) or _mode_from_filename(p.stem)
        if mode is None or mode not in range(1, 5):
            continue

        # For mode 3, enforce the PAG scale used in the cross-mode comparison
        if mode == 3:
            scale = _pag_scale_from_sidecar(meta)
            if scale is not None and abs(scale - pag_scale_for_comparison) > 1e-3:
                continue

        seed = _seed_from_sidecar(meta)
        if target_seed is not None and seed != target_seed:
            continue

        if mode not in mode_to_path:   # keep first (alphabetical)
            mode_to_path[mode] = p

    return mode_to_path


# =============================================================================
# Figure assembly
# =============================================================================

def build_figure(mode_images: dict[int, Path], item: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(
        f"Figure 4.2 — Attention Mode Comparison ({item})\n"
        "Same prompt · Same seed · Four attention configurations",
        fontsize=13, fontweight="bold", y=0.98,
    )

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, mode_id in enumerate(range(1, 5)):
        row, col = positions[idx]
        ax = axes[row][col]
        color = BORDER_COLORS[idx]

        if mode_id in mode_images:
            img  = Image.open(mode_images[mode_id]).convert("RGB")
            meta = _read_sidecar(mode_images[mode_id])
            seed_str = f"  seed={meta['Seed']}" if "Seed" in meta else ""
            ax.imshow(img)
            ax.set_title(
                f"{MODE_LABELS[mode_id]}{seed_str}",
                fontsize=11, fontweight="bold", pad=8, color=color,
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
        else:
            ax.set_facecolor("#f0f0f0")
            ax.text(
                0.5, 0.5,
                f"{MODE_LABELS[mode_id]}\n\n[Image not found]\n\n"
                "Generate images for this mode\nin app.py first.",
                ha="center", va="center", fontsize=10, color="#888888",
                transform=ax.transAxes,
            )
            ax.set_title(
                MODE_LABELS[mode_id],
                fontsize=11, fontweight="bold", pad=8, color=color,
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
                spine.set_linestyle("--")

        ax.set_xticks([])
        ax.set_yticks([])

    patches = [
        mpatches.Patch(color=BORDER_COLORS[i], label=MODE_LABELS[i + 1])
        for i in range(4)
    ]
    fig.legend(
        handles=patches, loc="lower center", ncol=2, fontsize=9,
        framealpha=0.9, bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Figure saved → {OUTPUT_PATH}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print(f"Item           : {ITEM}")
    print(f"Gen folder     : {GEN_DIR}  (prefix '{GEN_PREFIX}')")
    print(f"Target seed    : {TARGET_SEED if TARGET_SEED is not None else 'any (first found)'}")
    print(f"PAG scale used : {PAG_SCALE_FOR_COMPARISON} (for Mode 3 in cross-mode comparison)")

    mode_images = collect_mode_images(GEN_DIR, GEN_PREFIX, TARGET_SEED, PAG_SCALE_FOR_COMPARISON)

    print(f"\nFound images for modes: {sorted(mode_images.keys())}")
    for m, p in sorted(mode_images.items()):
        print(f"  Mode {m}: {p.name}")

    missing = [m for m in range(1, 5) if m not in mode_images]
    if missing:
        print(f"\n⚠  Missing modes: {missing} — placeholders will be shown.")

    build_figure(mode_images, ITEM)


if __name__ == "__main__":
    main()
