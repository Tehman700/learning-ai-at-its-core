"""
STEP 5 of 7: THE REVERSE PROCESS (generating new images)
========================================================
GOAL
    Start from pure random noise and remove noise step by step with the trained U-Net,
    until a brand-new image appears. No real photos are used here at all.

THE DDPM SAMPLING LOOP (from t = 999 down to 0):

    x = pure noise, torch.randn(n, C, H, W)
    for t = 999, 998, ..., 0:
        ε̂ = model(x, t)                                          ← "which part of this is noise?"
        x = 1/sqrt(α_t) * ( x - β_t / sqrt(1 - ᾱ_t) * ε̂ )        ← remove a little of that noise
        if t > 0:
            x = x + sqrt(β_t) * fresh_random_noise               ← add a tiny bit back (yes, really!)

    Why add noise back? Each step only removes a little noise, and the small random kick
    keeps the result varied and natural. Without it, DDPM images come out smooth and blurry.

FASTER SAMPLING: DDIM (used for the training previews and the app)
    1000 steps is slow. DDIM skips steps (e.g. 50 evenly spaced timesteps) with no retraining:
        x0_guess = (x - sqrt(1 - ᾱ_t) * ε̂) / sqrt(ᾱ_t)                ← "what the clean image probably is"
        x        = sqrt(ᾱ_prev) * x0_guess + sqrt(1 - ᾱ_prev) * ε̂      ← jump straight to the earlier timestep
    (ᾱ_prev is ᾱ at the next timestep in the shorter list; 1.0 after the last one)

SAVING THE RESULTS
    - a grid of generated images:  torchvision.utils.save_image
    - an animated GIF of pure noise turning into images, made with PIL (Pillow)

HOW TO RUN
    python step5_sample.py   (needs the checkpoint saved by STEP 4)

DONE WHEN
    [ ] samples/<checkpoint>_generated.png shows NEW digits (or dogs) that aren't copies of the training data
    [ ] samples/<checkpoint>_denoising.gif shows static turning into images
    [ ] DDIM with 50 steps gives similar images about 20x faster than DDPM

NEXT: step6_text_conditioning.py
"""
import time

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from config import (CHANNELS, DDIM_STEPS, DEVICE, GUIDANCE_SCALE, IMAGE_SIZE, NUM_CLASSES, NUM_SAMPLES,
                    SAMPLES_DIR, TIMESTEPS, USE_LABELS)
from step2_noise_schedule import make_schedule
from step3_unet import UNet


def load_model(path=None):
    """Build the U-Net and load the trained EMA weights. Returns (model, checkpoint)."""
    from step4_train import CHECKPOINT_PATH
    path = path or CHECKPOINT_PATH
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}. Train first with STEP 4 (or copy the file from Colab/Drive).")
    checkpoint = torch.load(path, map_location=DEVICE)
    model = UNet().to(DEVICE)
    model.load_state_dict(checkpoint["ema"])       # EMA weights give cleaner images
    model.eval()                                   # inference mode: dropout OFF
    print(f"loaded {path.name}: trained for {checkpoint['epoch']} epochs")
    return model, checkpoint


def predict_noise(model, x, t, labels=None, guidance=GUIDANCE_SCALE):
    """Ask the model for the noise. With labels, use classifier-free guidance (STEP 6)."""
    if labels is None:
        return model(x, t)
    # Ask twice in ONE batch (faster): "noise if it's this breed" and "noise for any image"
    no_label = torch.full_like(labels, NUM_CLASSES)
    both = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([labels, no_label]))
    with_label, without_label = both.chunk(2)
    # Push AWAY from "any image" and TOWARD "this breed"
    return without_label + guidance * (with_label - without_label)


def grid_picture(x):
    """A batch of images (-1..1) → one PIL picture (used for the GIF frames)."""
    return to_pil_image(make_grid(x.clamp(-1, 1).cpu(), nrow=4, normalize=True, value_range=(-1, 1)))


@torch.no_grad()
def ddpm_sample(model, schedule, n=NUM_SAMPLES, labels=None, guidance=GUIDANCE_SCALE, return_frames=False):
    """The original sampler: all 1000 steps. Slow but simple."""
    x = torch.randn(n, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)   # start: pure noise
    frames = []
    for t in tqdm(reversed(range(TIMESTEPS)), total=TIMESTEPS, desc="DDPM", leave=False):
        t_batch = torch.full((n,), t, device=DEVICE, dtype=torch.long)
        eps = predict_noise(model, x, t_batch, labels, guidance)

        beta, alpha, alpha_bar = schedule["betas"][t], schedule["alphas"][t], schedule["alpha_bars"][t]
        x = (x - beta / (1 - alpha_bar).sqrt() * eps) / alpha.sqrt()      # remove a little noise
        if t > 0:
            x = x + beta.sqrt() * torch.randn_like(x)                     # small random kick

        if return_frames and t % 20 == 0:
            frames.append(grid_picture(x))
    x = x.clamp(-1, 1)
    return (x, frames) if return_frames else x


@torch.no_grad()
def ddim_sample(model, schedule, n=NUM_SAMPLES, labels=None, guidance=GUIDANCE_SCALE, steps=DDIM_STEPS,
                return_frames=False):
    """The fast sampler: only `steps` timesteps (e.g. 999, 979, ..., 0). No randomness after the start."""
    timesteps = torch.linspace(TIMESTEPS - 1, 0, steps).long().tolist()
    x = torch.randn(n, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    frames = []
    for i, t in enumerate(timesteps):
        t_batch = torch.full((n,), t, device=DEVICE, dtype=torch.long)
        eps = predict_noise(model, x, t_batch, labels, guidance)

        alpha_bar = schedule["alpha_bars"][t]
        alpha_bar_prev = schedule["alpha_bars"][timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=DEVICE)

        x0_guess = ((x - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()).clamp(-1, 1)   # "the clean image is probably..."
        eps = (x - alpha_bar.sqrt() * x0_guess) / (1 - alpha_bar).sqrt()                 # noise that matches the clamped guess
        x = alpha_bar_prev.sqrt() * x0_guess + (1 - alpha_bar_prev).sqrt() * eps          # jump to the earlier timestep

        if return_frames:
            frames.append(grid_picture(x))
    x = x.clamp(-1, 1)
    return (x, frames) if return_frames else x


def preview_labels(n=NUM_SAMPLES):
    """Which labels to show in previews: 0, 1, 2, ... (MNIST: one of each digit; dogs: spread-out breeds)."""
    if not USE_LABELS:
        return None
    step = max(1, NUM_CLASSES // n)
    return (torch.arange(n, device=DEVICE) * step) % NUM_CLASSES


def save_preview(model, schedule, epoch, name_prefix="preview"):
    """Called by STEP 4 during training: generate a grid with the fast sampler and save it."""
    images = ddim_sample(model, schedule, labels=preview_labels())
    path = SAMPLES_DIR / f"{name_prefix}_epoch_{epoch:03d}.png"
    save_image(images, path, nrow=4, normalize=True, value_range=(-1, 1))
    return path


def save_gif(frames, path, size=256):
    """frames: list of PIL images → an animated GIF (the last frame stays on screen longer)."""
    frames = [frame.resize((size, size), resample=Image.NEAREST) for frame in frames]   # NEAREST keeps pixels sharp
    durations = [60] * (len(frames) - 1) + [2000]                                       # milliseconds per frame
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)


if __name__ == "__main__":
    torch.manual_seed(0)
    model, checkpoint = load_model()
    schedule = make_schedule()
    name = f"{checkpoint['dataset']}{'_labels' if USE_LABELS else ''}"
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    started = time.time()
    images, frames = ddpm_sample(model, schedule, labels=preview_labels(), return_frames=True)
    print(f"DDPM, {TIMESTEPS} steps: {time.time() - started:.1f}s")
    save_image(images, SAMPLES_DIR / f"{name}_generated.png", nrow=4, normalize=True, value_range=(-1, 1))
    save_gif(frames, SAMPLES_DIR / f"{name}_denoising.gif")

    started = time.time()
    images = ddim_sample(model, schedule, labels=preview_labels())
    print(f"DDIM, {DDIM_STEPS} steps: {time.time() - started:.1f}s")
    save_image(images, SAMPLES_DIR / f"{name}_generated_ddim.png", nrow=4, normalize=True, value_range=(-1, 1))
    print("saved", SAMPLES_DIR / f"{name}_generated.png", "+ _denoising.gif + _generated_ddim.png")
