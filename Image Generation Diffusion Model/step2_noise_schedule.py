"""
STEP 2 of 7: THE FORWARD PROCESS (turning an image into noise)
==============================================================
GOAL
    Write the maths that adds exactly the right amount of noise to an image for any step t.
    The U-Net (STEP 3) will later learn to UNDO this.

WHAT "NOISE" IS
    torch.randn_like(image) gives one random number per pixel, drawn from a bell curve
    ("Gaussian" / "normal" distribution): average 0, spread (standard deviation) 1.
    Most values are between -1 and 1, a few go out to ±3. On screen it looks like TV static.

THE IDEA IN PLAIN WORDS
    Imagine adding a tiny bit of TV static to a photo, 1000 times in a row.
    After 1000 times, the photo is gone: it is pure static.

    β_t ("beta")      how much NEW noise is added at step t. It starts tiny (0.0001) and grows to 0.02.
    α_t = 1 - β_t     how much of the image survives that ONE step
    ᾱ_t ("alpha bar") = α_0 × α_1 × ... × α_t   how much of the ORIGINAL image survives after t steps
                      (torch.cumprod computes this "running product" for all t at once)

THE SHORTCUT (why diffusion training is fast)
    We don't need to add noise 500 times to reach step 500. One formula jumps straight there:

        x_t = sqrt(ᾱ_t) * x0  +  sqrt(1 - ᾱ_t) * ε          where ε = random noise
              └─ how much image ┘   └─ how much noise ─┘

    t = 0   → ᾱ ≈ 1 → almost all image
    t = 999 → ᾱ ≈ 0 → almost all noise

    Why square roots? They keep the overall "strength" of x_t the same at every step
    (its spread stays about 1), so the U-Net always sees inputs on the same scale.
    The two parts always balance: (image part)² + (noise part)² = ᾱ + (1 - ᾱ) = 1.

HOW TO RUN
    python step2_noise_schedule.py
    → prints how much image vs noise there is at several timesteps
    → saves samples/step2_forward_process.png: images dissolving from t = 0 to t = 999
    → saves samples/step2_schedule.png: a chart of the image and noise amounts over time

DONE WHEN
    [ ] the image part goes from ~1.0 (t=0) down to ~0.0 (t=999)
    [ ] the picture strip shows the images slowly dissolving into static
    [ ] at t=999 the noisy images have mean ≈ 0 and std ≈ 1 (they are pure noise)

NEXT: step3_unet.py
"""
import torch

from config import BETA_END, BETA_START, DEVICE, SAMPLES_DIR, TIMESTEPS


def make_schedule(device=DEVICE):
    """Precompute every number we'll need, for all 1000 timesteps, once.

    Returns a dict of tensors, each with TIMESTEPS (1000) values: entry [t] belongs to timestep t.
    """
    betas = torch.linspace(BETA_START, BETA_END, TIMESTEPS)   # 1000 evenly spaced values: 0.0001 → 0.02
    alphas = 1.0 - betas                                       # how much survives ONE step
    alpha_bars = torch.cumprod(alphas, dim=0)                  # running product: how much of the ORIGINAL survives
    #   cumprod example:  [0.9, 0.8, 0.5] → [0.9, 0.9*0.8, 0.9*0.8*0.5] = [0.9, 0.72, 0.36]

    schedule = {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": alpha_bars.sqrt(),                  # the "how much image" number
        "sqrt_one_minus_alpha_bars": (1.0 - alpha_bars).sqrt(),  # the "how much noise" number
    }
    return {name: values.to(device) for name, values in schedule.items()}


def add_noise(x0, t, noise, schedule):
    """Jump straight to timestep t:  x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * noise

    x0:    clean images, shape (B, C, H, W), values -1..1
    t:     one timestep per image, shape (B,), whole numbers 0..TIMESTEPS-1
           (each image in a batch can have a DIFFERENT t; training uses this)
    noise: random noise, same shape as x0
    returns x_t, the noisy images, same shape as x0
    """
    image_amount = schedule["sqrt_alpha_bars"][t]              # one number per image → shape (B,)
    noise_amount = schedule["sqrt_one_minus_alpha_bars"][t]    # shape (B,)

    # (B,) → (B, 1, 1, 1) so that each image's number multiplies ALL of its pixels.
    # PyTorch "broadcasting" stretches the 1s to match (C, H, W) automatically.
    image_amount = image_amount.view(-1, 1, 1, 1)
    noise_amount = noise_amount.view(-1, 1, 1, 1)

    return image_amount * x0 + noise_amount * noise


# ─────────────────────────────────────────────────────────────────────────────
# Test and visualise
# ─────────────────────────────────────────────────────────────────────────────
def show_noise():
    print("===== WHAT NOISE LOOKS LIKE AS NUMBERS =====")
    noise = torch.randn(3, 64, 64)                             # the same shape as one colour dog image
    print(f"noise: mean {noise.mean():.3f}, std {noise.std():.3f}, min {noise.min():.2f}, max {noise.max():.2f}")
    print("first 6 values:", [round(v, 2) for v in noise.flatten()[:6].tolist()])
    share = (noise.abs() < 1).float().mean().item()
    print(f"{100 * share:.0f}% of the values are between -1 and 1 (the bell curve)\n")


def show_schedule(schedule):
    print("===== HOW MUCH IMAGE vs NOISE AT EACH TIMESTEP =====")
    print("   t   |  β_t    |   ᾱ_t    | image part sqrt(ᾱ) | noise part sqrt(1-ᾱ)")
    for t in [0, 50, 100, 250, 500, 750, 999]:
        print(f" {t:4d}  | {schedule['betas'][t].item():.5f} | {schedule['alpha_bars'][t].item():.6f} |"
              f"        {schedule['sqrt_alpha_bars'][t].item():.3f}       |       "
              f"{schedule['sqrt_one_minus_alpha_bars'][t].item():.3f}")

    # A chart of the two amounts over all 1000 steps
    import matplotlib
    matplotlib.use("Agg")                                      # draw to a file, no window needed
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    plt.plot(schedule["sqrt_alpha_bars"].cpu(), label="image part  sqrt(ᾱ_t)")
    plt.plot(schedule["sqrt_one_minus_alpha_bars"].cpu(), label="noise part  sqrt(1 - ᾱ_t)")
    plt.xlabel("timestep t")
    plt.ylabel("amount")
    plt.title("Forward process: the image fades out, the noise fades in")
    plt.legend()
    plt.grid(alpha=0.3)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(SAMPLES_DIR / "step2_schedule.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("saved", SAMPLES_DIR / "step2_schedule.png", "\n")


def show_forward_process(schedule):
    from torchvision.utils import save_image
    from step1_dataset import get_dataset

    print("===== IMAGES DISSOLVING INTO NOISE =====")
    dataset = get_dataset()
    picks = [0, len(dataset) // 4, len(dataset) // 2, 3 * len(dataset) // 4]   # 4 images spread across the dataset
    x0 = torch.stack([dataset[i][0] for i in picks]).to(DEVICE)       # 4 clean images, shape (4, C, H, W)
    timesteps = [0, 50, 100, 200, 300, 500, 700, 999]

    rows = []
    for image in x0:                                           # one row per image
        image = image.unsqueeze(0)                             # (C, H, W) → (1, C, H, W): a batch of one
        for t in timesteps:
            t_tensor = torch.tensor([t], device=DEVICE)
            x_t = add_noise(image, t_tensor, torch.randn_like(image), schedule)
            rows.append(x_t)
    grid = torch.cat(rows)                                     # (4 * 8, C, H, W)

    save_image(grid, SAMPLES_DIR / "step2_forward_process.png", nrow=len(timesteps),
               normalize=True, value_range=(-1, 1))            # anything outside -1..1 is clipped for display
    print("columns are t =", timesteps)
    print("saved", SAMPLES_DIR / "step2_forward_process.png")

    # Proof that t = 999 really is pure noise: its numbers look exactly like torch.randn's
    t_last = torch.full((4,), TIMESTEPS - 1, device=DEVICE)
    x_last = add_noise(x0, t_last, torch.randn_like(x0), schedule)
    print(f"at t = 999: mean {x_last.mean():.3f}, std {x_last.std():.3f}  (pure noise has mean 0, std 1)")
    print(f"clean images for comparison: mean {x0.mean():.3f}, std {x0.std():.3f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    schedule = make_schedule()
    show_noise()
    show_schedule(schedule)
    show_forward_process(schedule)
