import gradio as gr
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionPAGPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL
)
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0

import torch
import os
import time
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2

# Boost performance with cuDNN optimizations and TF32
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ================================
#  Model & Environment Setup
# ================================

# FIX #1: Use os.path.join instead of hardcoded backslashes for cross-platform compatibility
model_path = os.path.join("models", "Stable-diffusion", "dreamshaper_8.safetensors")

print("Loading base model...")

base_pipe = None
try:
    base_pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    print("✓ Base model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise e

base_pipe = base_pipe.to("cuda")
print("Model moved to CUDA device.")

# Load custom VAE
vae_path = os.path.join("models", "vae", "vaeKlF8Anime2_klF8Anime2VAE.safetensors")
vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float16)
base_pipe.vae = vae.to("cuda")

# FIX #9: Enable VAE slicing and tiling on base_pipe to prevent OOM at high resolutions
base_pipe.vae.enable_slicing()
base_pipe.vae.enable_tiling()
print("✓ Using kl-f8-anime2 VAE for maximum sharpness (slicing+tiling enabled)")

# DPMSolver++ Scheduler
base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    base_pipe.scheduler.config,
    solver_order=2,
    algorithm_type="dpmsolver++",
    use_karras_sigmas=True
)
print("✓ DPMSolver++ Scheduler applied")

# Enable xformers if available
try:
    base_pipe.enable_xformers_memory_efficient_attention()
    print("✓ xformers enabled")
except Exception:
    print("xformers not installed (optional)")

print("Torch compile disabled (Windows compatibility)")

# ================================
#  Load LoRA models
# ================================

lora_folder = os.path.join("models", "Lora")
lora_files = {}

if os.path.exists(lora_folder):
    for file in os.listdir(lora_folder):
        if file.endswith(".safetensors"):
            lora_files[file.replace(".safetensors", "")] = os.path.join(lora_folder, file)
            print(f"Detected LoRA: {file}")

# ================================
# Style & Item Prompts
# ================================

PROMPT_PREFIX = (
    "game prop, one weapon only, single object, solo item, "
    "isolated, centered composition, grey background, studio lighting,"
)

PROMPT_SUFFIX = (
    "masterpiece, best quality, highly detailed, "
    "weapon concept art, game asset render,"
)

item_prompts = {
    "Sword":
        "single sword, straight blade, decorated hilt, detailed pommel, full weapon,",

    "Bow":
        "single longbow, curved limbs, taut bowstring, ornamental riser, full weapon,",

    "Polearm":
        "single spear, one spearhead, leaf-shaped spearhead, ornate socket, decorative guard, octagonal hardwood shaft, brass fittings, engraved metalwork, full weapon,",

    "Armor":
        "full plate armor, complete armor set, breastplate, pauldrons, gauntlets, greaves, front view,",

    "Shield":
        "single heater shield, embossed surface, decorative border, reinforced metal rim, front facing,",
}

style_prompts = {
    "Ancient":
        "bronze and gold, engraved patterns, weathered metal, ornate craftsmanship, ancient relic,",

    "Modern":
        "matte black, carbon fiber, tactical design, clean geometry, gunmetal finish,",

    "Elemental":
        "glowing runes, magical energy, elemental crystal, arcane markings, mystical aura,",
}

NEGATIVE_PROMPT = (
    "human, multiple weapons, two weapons, paired weapons, duplicate, "
    "mirror, mirrored, mirror image, side by side, repeated object, "
    "double-ended, symmetrical composition, split image, collage, "
    "character, face, hands, body, holding, wearing, "
    "stand, pedestal, display base, mannequin, doll, statue, "
    "text, watermark, logo, blurry, lowres, cropped"
)

lora_mapping = {
    "Sword": "sword-000186",
    "Bow": "last_bow-000038",
    "Polearm": "polearm-000044",
    "Armor": "armor-000106",
    "Shield": "shield-000060",
}

# FIX #7: Use add_special_tokens=False to get the actual prompt token count,
#         excluding BOS/EOS special tokens which inflated the count by 2.
def _sd15_clip_token_count(text: str) -> int:
    """Return the number of CLIP tokens used by the prompt, excluding special tokens."""
    return len(base_pipe.tokenizer.encode(text, add_special_tokens=False))

# ================================
# Quantitative Metrics
# ================================

# FIX #8: Guard against None image to prevent crash when generation fails
def compute_image_metrics(image: Image.Image):
    """
    Compute image quality metrics: sharpness, edge density, and contrast.
    Focuses on the center crop to avoid bias from the plain background.
    Returns (0, 0, 0) if image is None.
    """
    if image is None:
        return 0.0, 0.0, 0.0

    img_gray = np.array(image.convert("L"))

    # Crop the center region to avoid background influence
    h, w = img_gray.shape
    crop = img_gray[h // 4:3 * h // 4, w // 4:3 * w // 4]

    # Sharpness via Laplacian variance
    laplacian = cv2.Laplacian(crop, cv2.CV_64F)
    sharpness = laplacian.var()

    # Edge density via Canny
    edges = cv2.Canny(crop, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Contrast via standard deviation
    contrast = crop.std()

    return sharpness, edge_density, contrast

# ================================
# PAG Pipeline Management
# ================================

# FIX #2 & #5: Manage the PAG pipeline in a dedicated function so it can be
#              rebuilt whenever the LoRA state changes.
_pag_pipe_cache = None
_pag_pipe_lora_state = None  # Tracks the LoRA state at the time the cache was created

def get_pag_pipe(lora_state_key: str):
    """
    Return a cached PAG pipeline, rebuilding it if the LoRA state has changed.

    Args:
        lora_state_key: A unique string representing the current LoRA configuration.

    Note:
        The PAG pipeline shares the same UNet as base_pipe (created via from_pipe),
        so LoRA weights loaded onto base_pipe are automatically reflected here.
        Rebuilding is only needed when from_pipe needs to re-wrap the updated pipe object.
    """
    global _pag_pipe_cache, _pag_pipe_lora_state

    if _pag_pipe_cache is None or _pag_pipe_lora_state != lora_state_key:
        print(f"Creating/Refreshing PAG pipeline (LoRA state: {lora_state_key})...")
        pag_pipe = StableDiffusionPAGPipeline.from_pipe(base_pipe)
        pag_pipe = pag_pipe.to("cuda")
        try:
            pag_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        pag_pipe.vae.enable_slicing()
        pag_pipe.vae.enable_tiling()
        _pag_pipe_cache = pag_pipe
        _pag_pipe_lora_state = lora_state_key
        print("PAG pipeline ready.")

    return _pag_pipe_cache

# ================================
# Image Generation Function
# ================================

def generate_item(style, item, use_lora_choice, attention_mode, pag_scale_slider, seed, use_random_seed):
    """
    Main generation function.

    Fixes applied:
    - LoRA residual guard: unload is called even if an exception occurs during loading
    - PAG pipeline is refreshed whenever the LoRA configuration changes
    - adaptive_callback uses a closure variable for total steps instead of unreliable pipe attributes
    - Token count excludes special tokens
    - Safe metrics computation handles None images
    - VAE slicing/tiling enabled on base_pipe to avoid OOM
    - Slider value is never set to None
    """
    try:
        torch.cuda.empty_cache()

        # ====================== 1. LoRA Handling ======================
        lora_strength = 0.9
        lora_status_line = "LoRA: No (disabled)"
        current_lora_key = "none"  # Used as cache key for PAG pipeline

        # FIX #4: Wrap LoRA loading in try/except so unload always runs on failure,
        #         preventing stale weights from leaking into subsequent calls.
        lora_loaded = False
        try:
            if use_lora_choice == "Yes" and item in lora_mapping and lora_mapping[item]:
                lora_key = lora_mapping[item]
                if lora_key in lora_files:
                    lora_path = lora_files[lora_key]
                    base_pipe.unload_lora_weights()
                    base_pipe.load_lora_weights(lora_path, adapter_name="default")
                    base_pipe.set_adapters(["default"], adapter_weights=[lora_strength])
                    lora_loaded = True
                    current_lora_key = f"{lora_key}_{lora_strength}"
                    lora_status_line = f"LoRA: Yes ({lora_key}) strength={lora_strength:.2f}"
                    print(f"Loaded LoRA: {lora_key} | strength={lora_strength:.2f}")
                else:
                    base_pipe.unload_lora_weights()
                    lora_status_line = f"LoRA: No (file missing: {lora_key})"
            else:
                base_pipe.unload_lora_weights()
        except Exception as lora_err:
            # Safe unload on failure to prevent weight accumulation
            try:
                base_pipe.unload_lora_weights()
            except Exception:
                pass
            lora_status_line = f"LoRA: Failed ({lora_err})"
            print(f"LoRA load error: {lora_err}")

        # ====================== 2. Build Prompt ======================
        item_text = item_prompts.get(item, item_prompts["Sword"])
        style_text = style_prompts.get(style, "")

        prompt = f"{PROMPT_PREFIX}{item_text}, {style_text}{PROMPT_SUFFIX}"
        negative_prompt = NEGATIVE_PROMPT

        # FIX #7: Count tokens without special tokens for accurate budget tracking
        token_count = _sd15_clip_token_count(prompt)
        print(f"Prompt ({token_count} tokens, excl. special): {prompt}")

        # ====================== 3. Seed & Generator ======================
        item_height = 768 if item in ["Armor", "Shield"] else 1024
        item_width = 768 if item in ["Armor", "Shield"] else 512

        if use_random_seed:
            seed_int = torch.randint(0, 2 ** 32 - 1, (1,)).item()
        else:
            try:
                seed_int = int(seed)
            except (ValueError, TypeError):
                seed_int = torch.randint(0, 2 ** 32 - 1, (1,)).item()

        generator = torch.Generator("cuda").manual_seed(seed_int)
        seed_used = str(seed_int)
        print(f"Using seed: {seed_used}")

        # ====================== 4. Pipeline Selection & Generation ======================
        start_time = time.time()
        mode = attention_mode.strip()

        if mode in ["PAG Enhanced", "Adaptive PAG + Progressive CFG", "Adaptive PAG + CFG"]:

            # FIX #2: Rebuild PAG pipeline if LoRA state has changed since last call
            pipe = get_pag_pipe(current_lora_key)

            if mode in ["Adaptive PAG + Progressive CFG", "Adaptive PAG + CFG"]:

                # ====================== Adaptive PAG + Progressive CFG ======================
                attn_status = "Attention: Adaptive PAG + Progressive CFG (Proposed method)"

                # FIX #2: Use a closure variable for total steps instead of querying
                #         unreliable pipe attributes (which often fall back to a hardcoded default anyway)
                num_steps = 40

                def adaptive_callback(pipe_ref, i, t, callback_kwargs):
                    """Dynamically scale CFG and PAG guidance throughout denoising."""
                    progress = float(i) / num_steps

                    # Both scales ramp up progressively: low early (preserve structure),
                    # high late (sharpen details)
                    cfg_scale = 5.5 + 3.5 * (progress ** 1.65)
                    pag_scale = 0.8 + 2.7 * (progress ** 1.75)

                    pipe_ref._guidance_scale = cfg_scale
                    if hasattr(pipe_ref, "_pag_scale"):
                        pipe_ref._pag_scale = pag_scale

                    return callback_kwargs

                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_steps,
                    height=item_height,
                    width=item_width,
                    guidance_scale=6.0,
                    generator=generator,
                    pag_scale=1.5,
                    callback_on_step_end=adaptive_callback,
                )
                image = output.images[0]

            else:
                # ====================== PAG Enhanced (fixed scale) ======================
                attn_status = f"Attention: PAG Enhanced (scale={pag_scale_slider:.1f})"
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=35,
                    height=item_height,
                    width=item_width,
                    guidance_scale=7.0,
                    generator=generator,
                    pag_scale=pag_scale_slider,
                )
                image = output.images[0]

        else:
            pipe = base_pipe

            if attention_mode == "Naive (Baseline)":
                if not lora_loaded:
                    pipe.unet.set_attn_processor(AttnProcessor())
                attn_status = "Attention: Naive (Baseline)"
            else:
                if not lora_loaded:
                    pipe.unet.set_attn_processor(AttnProcessor2_0())
                attn_status = "Attention: Default (AttnProcessor2_0)"

            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=35,
                height=item_height,
                width=item_width,
                guidance_scale=7.0,
                generator=generator,
            )
            image = output.images[0]

        # ====================== 5. Post-processing & Metrics ======================
        image = image.filter(ImageFilter.UnsharpMask(radius=0.8, percent=35, threshold=2))
        image = ImageEnhance.Sharpness(image).enhance(1.08)

        gen_time = time.time() - start_time

        # FIX #8: compute_image_metrics handles None internally; safe to call unconditionally
        sharpness, edge_density, contrast = compute_image_metrics(image)
        
        print(f"PREFIX tokens:  {_sd15_clip_token_count(PROMPT_PREFIX)}")
        print(f"ITEM tokens:    {_sd15_clip_token_count(item_text)}")
        print(f"STYLE tokens:   {_sd15_clip_token_count(style_text)}")
        print(f"SUFFIX tokens:  {_sd15_clip_token_count(PROMPT_SUFFIX)}")
        print(f"TOTAL tokens:   {_sd15_clip_token_count(prompt)}")

        status = (
            f"✓ Generated {item} in {style} style.\n"
            f"{lora_status_line}\n"
            f"{attn_status}\n"
            f"Seed: {seed_used}\n"
            f"Time: {gen_time:.2f}s\n"
            f"Sharpness: {sharpness:.1f}\n"
            f"Edge Density: {edge_density:.4f}\n"
            f"Contrast: {contrast:.1f}"
        )

        # ====================== 6. Save Results ======================
        try:
            save_dir = os.path.join("output", "kd_gen")
            os.makedirs(save_dir, exist_ok=True)
            filename_base = (
                f"{item}_{style}_{attention_mode.replace(' ', '_')}"
                f"_{seed_used}_{int(time.time())}"
            )
            image_path = os.path.join(save_dir, f"{filename_base}.png")
            text_path = os.path.join(save_dir, f"{filename_base}.txt")

            image.save(image_path)
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(status + f"\n\nFull Prompt: {prompt}\nTokens: {token_count}")
            print(f"Saved: {image_path}")
        except Exception as e:
            print(f"Save warning: {e}")

        return image, status

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return None, f"Error: {str(e)}"


# ================================
#  Gradio Interface
# ================================

with gr.Blocks(title="Weapon & Item Generator") as demo:
    gr.Markdown("# ⚔️ Weapon & Item Generator")
    gr.Markdown("Attention modes with real PAG + quantitative metrics (Sharpness & Structure).")

    with gr.Row():
        with gr.Column():
            style_choice = gr.Radio(
                choices=["Ancient", "Modern", "Elemental"],
                value="Ancient",
                label="Select Style"
            )

        with gr.Column():
            item_choice = gr.Radio(
                choices=["Sword", "Bow", "Polearm", "Armor", "Shield"],
                value="Sword",
                label="Select Item"
            )
            use_lora = gr.Radio(
                choices=["Yes", "No"],
                value="Yes",
                label="Use LoRA"
            )

    attention_mode = gr.Radio(
        choices=["Default (AttnProcessor2_0)", "Naive (Baseline)", "PAG Enhanced", "Adaptive PAG + Progressive CFG"],
        value="Default (AttnProcessor2_0)",
        label="Attention Mode"
    )

    # FIX #6: Never pass None as the slider value; only toggle the interactive state.
    #         The slider stays at 2.5 when non-PAG modes are selected.
    pag_scale = gr.Slider(
        minimum=0.5,
        maximum=5.0,
        value=2.5,
        step=0.1,
        label="PAG Scale (only works in PAG Enhanced mode)",
        interactive=False
    )

    def update_pag_slider(x):
        """Enable the PAG scale slider only when PAG Enhanced mode is selected."""
        return gr.update(interactive=(x == "PAG Enhanced"))

    attention_mode.change(
        fn=update_pag_slider,
        inputs=attention_mode,
        outputs=pag_scale
    )

    seed_input = gr.Number(
        label="Seed (for reproducible comparison)",
        value=1333,
        precision=0,
        minimum=0
    )

    random_seed_checkbox = gr.Checkbox(
        label="Use Random Seed",
        value=False
    )

    generate_btn = gr.Button("🎨 Generate", size="lg")

    with gr.Row():
        output_image = gr.Image(label="Generated Item")
        status_box = gr.Textbox(
            label="Status + Metrics",
            interactive=False,
            lines=9,
        )

    generate_btn.click(
        fn=generate_item,
        inputs=[style_choice, item_choice, use_lora, attention_mode, pag_scale, seed_input, random_seed_checkbox],
        outputs=[output_image, status_box]
    )

demo.launch()