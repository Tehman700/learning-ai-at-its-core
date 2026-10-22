"""
STEP 4 of 7: TRAINING
=====================
GOAL
    Teach the U-Net to predict the noise that was added to an image.

THE WHOLE ALGORITHM (from the DDPM paper, Ho et al. 2020); it really is this short:

    repeat:
        x0    = a batch of real images                       (STEP 1)
        t     = a random timestep for each image, 0..999
        ε     = random noise, same shape as x0
        x_t   = add_noise(x0, t, ε)                          (STEP 2)
        ε̂     = model(x_t, t)                                (STEP 3)
        loss  = mean((ε - ε̂)²)        ← "how far was the guess from the real noise?"  (MSE)
        backward + optimizer step                            (the same loop as STEP 0!)

    Why a RANDOM t each time? So the model practises every noise level: barely noisy,
    half noisy, almost pure noise. At generation time it needs to handle all of them.

EXTRAS THAT MAKE IT WORK WELL
    - AMP (USE_AMP): mixed precision. The GPU does most maths in 16-bit instead of 32-bit numbers:
      about 2x faster on a T4 and uses less memory. GradScaler protects tiny gradients from
      rounding to zero in 16-bit.
    - Gradient clipping: if one unlucky batch produces a huge gradient, shrink it, so a single
      bad step can't wreck the weights.
    - EMA (EMA_DECAY): keep a slowly-updated copy of the weights:  ema = 0.999*ema + 0.001*current.
      It averages out the jitter of the last ~1000 steps. Sampling with the EMA copy gives
      noticeably cleaner images, so the checkpoint's "ema" weights are the ones we generate with.
    - Checkpoints every SAVE_EVERY epochs: Colab can disconnect, so we save often and RESUME
      automatically: run the script again and it continues from the last checkpoint.
    - A preview grid every SAVE_EVERY epochs (uses STEP 5): watching the images improve is the
      best progress bar. A low loss alone does not prove the images are good.

HOW TO RUN
    Quick test on a laptop (2 tiny epochs of 5 batches, checks that everything works, ~2–4 min on a CPU):
        python step4_train.py --quick
    Milestone 1, DATASET = "mnist": on Colab with train_on_colab.ipynb   (~15–30 min)
    Milestone 2, DATASET = "dogs":  on Colab with train_on_colab.ipynb   (a few hours)

WHAT TO EXPECT
    The loss starts around 1.0 and drops fast, then slowly (MNIST settles around ~0.03,
    dogs higher; your numbers will differ a bit). The preview grids go from static → blobs →
    recognisable digits (or dogs).

DONE WHEN
    [ ] --quick runs without errors and saves a checkpoint + preview
    [ ] on Colab, the loss goes down and the previews in samples/ improve epoch by epoch

NEXT: step5_sample.py
"""
import copy
import os
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import (CFG_DROP_PROB, CHECKPOINT_DIR, DATASET, DEVICE, EMA_DECAY, EPOCHS, LEARNING_RATE,
                    NUM_CLASSES, SAMPLES_DIR, SAVE_EVERY, TIMESTEPS, USE_AMP, USE_LABELS)
from step1_dataset import get_class_names, get_dataloader, get_dataset
from step2_noise_schedule import add_noise, make_schedule
from step3_unet import UNet

# One checkpoint file per setup, so MNIST and dogs (and with/without labels) never overwrite each other:
#   mnist.pt · dogs.pt · dogs_labels.pt
CHECKPOINT_PATH = CHECKPOINT_DIR / f"{DATASET}{'_labels' if USE_LABELS else ''}.pt"


@torch.no_grad()
def update_ema(ema_model, model, decay=EMA_DECAY):
    """ema = decay * ema + (1 - decay) * current   for every weight."""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param, alpha=1 - decay)


def save_checkpoint(path, model, ema_model, optimizer, scaler, epoch, class_names, losses):
    """Save everything needed to RESUME training or to GENERATE images later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),          # the weights being trained
        "ema": ema_model.state_dict(),        # the smoothed weights → used for generating
        "optimizer": optimizer.state_dict(),  # Adam's memory of past gradients (needed to resume smoothly)
        "scaler": scaler.state_dict(),        # AMP's scaling state
        "epoch": epoch,                       # how many epochs are done
        "class_names": class_names,           # label number → name (STEP 6 and the app need these)
        "losses": losses,                     # average loss of every epoch so far
        "dataset": DATASET,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)         # write to a temporary file first, then swap it in:
    os.replace(temporary, path)               # if Colab dies mid-save, the old checkpoint survives


def save_loss_curve(losses, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 3.5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", markersize=3)
    plt.xlabel("epoch")
    plt.ylabel("average loss (MSE)")
    plt.title(f"Training loss ({DATASET})")
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()


def train(epochs=EPOCHS, max_batches=None, save_every=SAVE_EVERY, checkpoint_path=CHECKPOINT_PATH):
    from step5_sample import save_preview     # imported here to avoid a circular import

    # ── SETUP ────────────────────────────────────────────────────────────────
    dataset = get_dataset()
    class_names = get_class_names(dataset)
    loader = get_dataloader(dataset)
    schedule = make_schedule()

    model = UNet().to(DEVICE)
    ema_model = copy.deepcopy(model).eval()                 # the EMA copy: never trained directly
    ema_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    use_amp = USE_AMP and DEVICE == "cuda"                  # AMP only helps on a GPU
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, losses = 0, []
    if checkpoint_path.exists():                            # RESUME after a disconnect
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, losses = checkpoint["epoch"], checkpoint["losses"]
        print(f"resuming from {checkpoint_path.name}: {start_epoch} epochs already done")
        if start_epoch >= epochs:
            print(f"nothing to do: all {epochs} epochs are finished (raise EPOCHS in config.py to train longer)")
            return model, ema_model, losses

    batches_per_epoch = min(len(loader), max_batches or len(loader))
    print(f"training on {DEVICE} | {DATASET}: {len(dataset):,} images | {batches_per_epoch} batches per epoch"
          f" | epochs {start_epoch + 1}→{epochs} | AMP {'on' if use_amp else 'off'} | labels {'on' if USE_LABELS else 'off'}")

    # ── THE LOOP ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()                                       # dropout ON while training
        total, count, started = 0.0, 0, time.time()
        progress = tqdm(loader, total=batches_per_epoch, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for batch_number, (x0, labels) in enumerate(progress):
            if max_batches and batch_number >= max_batches:
                break
            x0 = x0.to(DEVICE, non_blocking=True)
            batch = x0.shape[0]

            # 1. Make the exercise: a random noise level per image, random noise, mix (STEP 2)
            t = torch.randint(0, TIMESTEPS, (batch,), device=DEVICE)
            noise = torch.randn_like(x0)
            x_t = add_noise(x0, t, noise, schedule)

            # (STEP 6) Labels: hide them CFG_DROP_PROB of the time by swapping in the "no label" id,
            # so the same model also learns to draw "any dog" (needed for classifier-free guidance)
            if USE_LABELS:
                labels = labels.to(DEVICE)
                hide = torch.rand(batch, device=DEVICE) < CFG_DROP_PROB
                labels = torch.where(hide, torch.full_like(labels, NUM_CLASSES), labels)
            else:
                labels = None

            # 2. Forward + loss: guess the noise, measure how wrong the guess is
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                predicted_noise = model(x_t, t, labels)
            loss = F.mse_loss(predicted_noise.float(), noise)

            # 3. Backward + update (the same as STEP 0, plus AMP scaling and clipping)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                                  # back to normal size before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_model, model)

            total += loss.item()
            count += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        losses.append(total / count)
        print(f"epoch {epoch + 1:3d}/{epochs} | average loss {losses[-1]:.4f} | {time.time() - started:.0f}s")

        # 4. Every few epochs: save everything + a preview of what the model can draw now
        if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
            save_checkpoint(checkpoint_path, model, ema_model, optimizer, scaler, epoch + 1, class_names, losses)
            SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
            preview = save_preview(ema_model, schedule, epoch + 1, name_prefix=checkpoint_path.stem)
            save_loss_curve(losses, SAMPLES_DIR / f"{checkpoint_path.stem}_loss_curve.png")
            print(f"  saved {checkpoint_path.name} and preview {preview.name}")

    print("training finished")
    return model, ema_model, losses


if __name__ == "__main__":
    if "--quick" in sys.argv:
        # A smoke test: 2 tiny epochs of 5 batches, saved to a separate file so it never
        # gets mixed up with real training.
        train(epochs=2, max_batches=5, save_every=1,
              checkpoint_path=CHECKPOINT_DIR / f"{CHECKPOINT_PATH.stem}_quicktest.pt")
    else:
        train()
