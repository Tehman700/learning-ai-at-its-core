"""
config.py: ALL SETTINGS IN ONE PLACE
====================================
Every step imports its numbers from here. To change something (image size,
number of noise steps, learning rate), change it ONCE here.

You don't need to understand every value yet. Each one is explained again in the
step that uses it (look for the "used in STEP N" notes).
"""
from pathlib import Path

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"                # datasets (ignored by git: too big)
CHECKPOINT_DIR = ROOT / "checkpoints"   # saved weights (ignored by git). On Colab the notebook points this at Google Drive.
SAMPLES_DIR = ROOT / "samples"          # generated images and videos

# ─────────────────────────────────────────────────────────────────────────────
# Device: use the GPU if there is one (on Colab: Runtime → Change runtime type → T4 GPU)
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Dataset (used in STEP 1)
#   "mnist": Milestone 1. Handwritten digits, grayscale, trains in minutes.
#   "dogs":  Milestone 2+. Stanford Dogs, 120 breeds, colour, trains in hours.
# ─────────────────────────────────────────────────────────────────────────────
DATASET = "mnist"

if DATASET == "mnist":
    IMAGE_SIZE = 32        # MNIST is 28x28; we pad to 32 so the U-Net can halve it cleanly: 32 → 16 → 8
    CHANNELS = 1           # grayscale
    NUM_CLASSES = 10       # digits 0–9 (used in STEP 6)
else:
    IMAGE_SIZE = 64        # 64x64 colour images: small enough for a free GPU
    CHANNELS = 3           # RGB
    NUM_CLASSES = 120      # dog breeds (used in STEP 6)

# ─────────────────────────────────────────────────────────────────────────────
# Diffusion process (used in STEP 2)
# ─────────────────────────────────────────────────────────────────────────────
TIMESTEPS = 1000           # T: number of small noise steps from a clean image to pure noise
BETA_START = 1e-4          # β at the first step: a tiny bit of noise
BETA_END = 0.02            # β at the last step: more noise per step

# ─────────────────────────────────────────────────────────────────────────────
# U-Net size (used in STEP 3)
# ─────────────────────────────────────────────────────────────────────────────
BASE_CHANNELS = 64                         # width of the first layer; bigger = better images but slower
CHANNEL_MULTS = (1, 2, 2) if DATASET == "mnist" else (1, 2, 2, 4)
#   one entry per resolution level, e.g. dogs: 64px×64ch → 32px×128ch → 16px×128ch → 8px×256ch
TIME_EMB_DIM = 256                         # size of the vector that tells the U-Net which timestep t it's at

# ─────────────────────────────────────────────────────────────────────────────
# Training (used in STEP 4)
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 128 if DATASET == "mnist" else 64
LEARNING_RATE = 2e-4
EPOCHS = 20 if DATASET == "mnist" else 100  # dogs: watch the preview images and stop when they look good
USE_AMP = True             # mixed precision: ~2x faster on Colab GPUs, uses less memory
EMA_DECAY = 0.999          # keep a smoothed copy of the weights; it gives cleaner images (optional)
SAVE_EVERY = 5             # save a checkpoint and a preview grid every N epochs

# ─────────────────────────────────────────────────────────────────────────────
# Conditioning: "draw THIS breed" (used in STEP 6)
# ─────────────────────────────────────────────────────────────────────────────
USE_LABELS = False         # turn on in STEP 6
CFG_DROP_PROB = 0.1        # in training, hide the label 10% of the time so the model also learns "any dog"
GUIDANCE_SCALE = 4.0       # at generation: 1 = ignore the guidance trick, higher = follow the label more strongly

# ─────────────────────────────────────────────────────────────────────────────
# Sampling (used in STEP 5)
# ─────────────────────────────────────────────────────────────────────────────
NUM_SAMPLES = 16           # images per preview grid (4x4)
DDIM_STEPS = 50            # the fast sampler uses 50 steps instead of 1000 (optional part of STEP 5)
