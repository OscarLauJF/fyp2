FYP2 — WEAPON & ITEM GENERATOR
Installation and Execution Guide
================================

1. PROJECT INFORMATION
----------------------
Project title: Weapon & Item Generator
GitHub repository: https://github.com/OscarLauJF/fyp2
Developer: Lau Jia Fu
Student ID: 242UT243V1
Email: laujiafu5308@gmail.com
Submission version: v1.0-submission
Last updated: 13 July 2026


2. PROJECT OVERVIEW
-------------------
This project is a Gradio-based Stable Diffusion image-generation application
for producing game weapon and equipment concept art.

Supported item categories:
- Sword
- Bow
- Polearm
- Armor
- Shield

Supported visual styles:
- Ancient
- Modern
- Elemental

Supported attention and guidance modes:
- Default Attention (AttnProcessor2_0)
- Naive Attention Baseline
- PAG Enhanced
- Adaptive PAG + Progressive CFG

Each successful generation records the selected item and style, LoRA status,
attention mode, seed, inference time, sharpness, edge density, contrast, full
prompt, and prompt token count.

Generated PNG images and TXT metadata files are saved in:

    output/kd_gen/


3. SOURCE CODE
--------------
The latest source code is included directly in this GitHub repository:

    https://github.com/OscarLauJF/fyp2

Clone the repository:

    git clone https://github.com/OscarLauJF/fyp2.git
    cd fyp2

Alternatively, open the repository, click "Code", select "Download ZIP", and
extract the archive.

The repository contains:

    app.py
    evaluate_metrics.py
    figure4_2_attention_mode_comparison.py
    figure4_4_pag_scale_ablation.py
    table4_2_attention_mode_metrics.py
    table4_3_lora_ablation.py
    requirements.txt
    Readme.txt
    .gitignore


4. DATASET AND DATABASE REQUIREMENTS
------------------------------------
No database is required.

No dataset is required to install or execute app.py.

The application is an inference-only image-generation system. It uses a
pretrained Stable Diffusion checkpoint, a VAE, and optional LoRA weight files.

The evaluation scripts are auxiliary development scripts and are not required
to launch or use the Gradio application. Some of them refer to local reference
image folders when reproducing FID or SSIM experiments. Those local reference
images are not part of the normal installation requirements and are not needed
for application execution.


5. REQUIRED MODEL DOWNLOADS
---------------------------
Large model files are not included in the GitHub repository.

5.1 Base model

Source:

    https://huggingface.co/digiplay/DreamShaper_8/tree/main

Download the Stable Diffusion 1.5-compatible DreamShaper 8 checkpoint.

Required local filename:

    dreamshaper_8.safetensors

Required path:

    models/Stable-diffusion/dreamshaper_8.safetensors

If the downloaded file has a different filename, rename it or update model_path
in app.py. Do not use DreamShaper XL or DreamShaper LCM.

5.2 VAE

Source:

    https://civitai.com/models/23906/vae-kl-f8-anime2

This project uses the kl-f8-anime2 VAE.

Required local filename:

    vaeKlF8Anime2_klF8Anime2VAE.safetensors

Required path:

    models/vae/vaeKlF8Anime2_klF8Anime2VAE.safetensors

If the downloaded file has a different filename, rename it or update vae_path
in app.py.

5.3 LoRA models

The trained LoRA files are provided through Hugging Face:

    https://huggingface.co/oscar11112/fyp2

Download:

    sword-000186.safetensors
    last_bow-000038.safetensors
    polearm-000044.safetensors
    armor-000106.safetensors
    shield-000060.safetensors

Place them in:

    models/Lora/

The application can run without the LoRA files when "Use LoRA" is set to "No".
The base model and VAE are mandatory.


6. REQUIRED DIRECTORY STRUCTURE
-------------------------------
Arrange the project as follows:

    fyp2/
    |
    |-- app.py
    |-- evaluate_metrics.py
    |-- figure4_2_attention_mode_comparison.py
    |-- figure4_4_pag_scale_ablation.py
    |-- table4_2_attention_mode_metrics.py
    |-- table4_3_lora_ablation.py
    |-- requirements.txt
    |-- Readme.txt
    |-- .gitignore
    |
    |-- models/
    |   |-- Stable-diffusion/
    |   |   `-- dreamshaper_8.safetensors
    |   |-- vae/
    |   |   `-- vaeKlF8Anime2_klF8Anime2VAE.safetensors
    |   `-- Lora/
    |       |-- sword-000186.safetensors
    |       |-- last_bow-000038.safetensors
    |       |-- polearm-000044.safetensors
    |       |-- armor-000106.safetensors
    |       `-- shield-000060.safetensors
    |
    `-- output/
        |-- kd_gen/
        |-- figures/
        `-- tables/

The output directories are created automatically when the relevant scripts are
executed.


7. RECOMMENDED HARDWARE
-----------------------
Required:
- 64-bit Windows 10/11 or Linux
- NVIDIA CUDA-capable GPU
- Current NVIDIA graphics driver
- At least 8 GB GPU VRAM
- At least 16 GB system RAM
- At least 15 GB free storage

Recommended:
- NVIDIA GPU with 12 GB or more VRAM

Important:
app.py explicitly moves the Stable Diffusion pipeline and VAE to "cuda". The
current version will not run on a CPU-only computer, AMD GPU, or Apple Silicon
computer without source-code modifications.


8. SOFTWARE VERSIONS
--------------------
Recommended environment:
- Python 3.10.x
- PyTorch 2.5.1
- Torchvision 0.20.1
- CUDA-enabled PyTorch wheel: CUDA 12.1
- Diffusers 0.32.2
- Transformers 4.48.3
- Accelerate 1.3.0
- Safetensors 0.5.2
- Hugging Face Hub 0.28.1
- Gradio 5.17.1
- Pillow 11.1.0
- NumPy 1.26.4
- OpenCV Python 4.11.0.86
- SciPy 1.15.1
- scikit-image 0.25.1
- Matplotlib 3.10.0
- xFormers 0.0.29.post1 (optional)
- LPIPS 0.1.4 (optional)

Official pages:
- Python: https://www.python.org/downloads/
- Git: https://git-scm.com/downloads
- NVIDIA driver: https://www.nvidia.com/Download/index.aspx
- PyTorch: https://pytorch.org/get-started/locally/
- Diffusers: https://huggingface.co/docs/diffusers/installation
- Gradio: https://www.gradio.app/guides


9. WINDOWS INSTALLATION
-----------------------
1. Clone the repository:

       git clone https://github.com/OscarLauJF/fyp2.git
       cd fyp2

2. Create a virtual environment:

       py -3.10 -m venv .venv

3. Activate it in Command Prompt:

       .venv\Scripts\activate

   Or in PowerShell:

       .\.venv\Scripts\Activate.ps1

4. Upgrade pip:

       python -m pip install --upgrade pip setuptools wheel

5. Install CUDA-enabled PyTorch and Torchvision:

       pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

6. Install the remaining dependencies:

       pip install -r requirements.txt

7. Optional xFormers installation:

       pip install xformers==0.0.29.post1

8. Optional LPIPS installation:

       pip install lpips==0.1.4

9. Download and arrange the base model, VAE, and LoRA files according to
   Sections 5 and 6.

10. Verify CUDA:

       python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"

Expected result:

    CUDA available: True

The .venv directory is local and must not be uploaded to GitHub.


10. LINUX INSTALLATION
----------------------
Clone and enter the repository:

    git clone https://github.com/OscarLauJF/fyp2.git
    cd fyp2

Create and activate a virtual environment:

    python3.10 -m venv .venv
    source .venv/bin/activate

Upgrade pip:

    python -m pip install --upgrade pip setuptools wheel

Install CUDA-enabled PyTorch and Torchvision:

    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

Install the remaining dependencies:

    pip install -r requirements.txt

Optional dependencies:

    pip install xformers==0.0.29.post1
    pip install lpips==0.1.4

Download and arrange the base model, VAE, and LoRA files according to Sections
5 and 6.


11. RUNNING THE APPLICATION
---------------------------
Activate the virtual environment and run:

    python app.py

Wait until the base model and VAE have finished loading.

Open the local Gradio URL shown in the terminal. The default address is normally:

    http://127.0.0.1:7860

Usage:
1. Select a visual style.
2. Select an item category.
3. Enable or disable LoRA.
4. Select an attention mode.
5. Adjust the PAG scale when PAG Enhanced is selected.
6. Enter a fixed seed or enable random seed.
7. Click "Generate".
8. Wait for the generated image and metrics.
9. Check output/kd_gen/ for the PNG and TXT files.


12. OUTPUT FILES
----------------
For every successful generation, app.py saves:
- One PNG image
- One TXT metadata sidecar

The TXT file records:
- Selected item and style
- LoRA status
- Attention mode
- Seed
- Inference time
- Sharpness
- Edge density
- Contrast
- Full prompt
- Prompt token count

Output location:

    output/kd_gen/


13. OPTIONAL EVALUATION SCRIPTS
-------------------------------
The following scripts were used during project development:

    evaluate_metrics.py
    figure4_2_attention_mode_comparison.py
    figure4_4_pag_scale_ablation.py
    table4_2_attention_mode_metrics.py
    table4_3_lora_ablation.py

They are not required to execute app.py.

Some scripts use local reference images to reproduce FID or SSIM experiments.
Those reference images are not required for normal application use.

Run examples:

    python evaluate_metrics.py
    python figure4_2_attention_mode_comparison.py
    python figure4_4_pag_scale_ablation.py
    python table4_2_attention_mode_metrics.py
    python table4_3_lora_ablation.py

Possible outputs:

    output/figures/figure4_2_attention_mode_comparison.png
    output/figures/figure4_4_pag_scale_ablation.png
    output/tables/table4_2_attention_mode_metrics.csv
    output/tables/table4_3_lora_ablation.csv

During the first evaluation run, the scripts may automatically download
Inception V3, CLIP, or LPIPS weights.


14. TROUBLESHOOTING
-------------------
CUDA unavailable:
- Confirm that an NVIDIA GPU is installed.
- Update the NVIDIA graphics driver.
- Install the CUDA-enabled PyTorch build.

Base model not found:

    models/Stable-diffusion/dreamshaper_8.safetensors

VAE not found:

    models/vae/vaeKlF8Anime2_klF8Anime2VAE.safetensors

LoRA missing:
- Download all five LoRA files from Hugging Face.
- Place them in models/Lora/.
- Confirm that their filenames match Section 5.3.

CUDA out of memory:
- Close other GPU-intensive applications.
- Install xFormers if compatible.
- Restart Python.
- Reduce generation resolution or inference steps if necessary.

Gradio page does not open:
- Copy the URL printed in the terminal into a browser.
- Check whether port 7860 is already in use.
- Allow Python through the firewall if prompted.


15. GITHUB EXCLUSIONS
---------------------
Recommended .gitignore:

    .venv/
    venv/
    env/
    __pycache__/
    *.pyc
    models/
    data/
    output/
    .idea/
    .vscode/
    .DS_Store
    Thumbs.db

Do not upload:
- .venv
- Downloaded base model
- Downloaded VAE
- Generated output files
- Local evaluation reference images


16. IMPORTANT NOTES
-------------------
- No database is required.
- No dataset is required to execute app.py.
- The base model and VAE must be downloaded separately.
- The five LoRA files are provided through Hugging Face.
- The Stable Diffusion safety checker is disabled in app.py.
- Use the application responsibly.
- Follow the licences and usage conditions of the base model, VAE, LoRA files,
  and generated content.
- Keep the GitHub and Hugging Face links available for at least one year after
  submission.
