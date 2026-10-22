"""
STEP 3 of 7: THE U-NET (the network that predicts noise)
========================================================
GOAL
    Build the neural network:  (noisy image x_t, timestep t)  →  predicted noise ε̂
    Its output has EXACTLY the same shape as its input image: one noise guess per pixel.
    Here we only BUILD it. Its weights are random; STEP 4 trains them.

WHY A "U"-NET?
    input 64x64 ─► 32x32 ─► 16x16 ─► 8x8      (DOWN: shrink the image, learn "what is in it")
                                      │
                                   MIDDLE      (smallest size: sees the whole dog at once)
                                      │
    output 64x64 ◄─ 32x32 ◄─ 16x16 ◄─ 8x8     (UP: grow it back, rebuild the details)
          ▲         ▲         ▲
          └─────────┴─────────┴── SKIP CONNECTIONS: each level of the UP path also receives
                                  the matching level from the DOWN path, so fine details aren't lost.

    Drawn as a diagram, that is the shape of a "U". Stable Diffusion uses a (much bigger) U-Net too.

    Why shrink at all? A 3x3 convolution only sees 3x3 pixels. After shrinking 64 → 8, each pixel
    stands for an 8x8 patch, so the same 3x3 convolution now "sees" a big part of the dog.
    The DOWN path understands the big picture; the UP path + skips put the fine detail back.

THE BUILDING BLOCKS (in the order they appear below)
    1. SinusoidalTimeEmbedding: turns the number t into a vector, so the network knows
                                 how noisy the input is (t=10 needs gentle cleaning, t=900 a lot).
                                 It's the same trick transformers use for word positions.
    2. ResBlock:                 two convolutions + "add the input back" (residual), with the
                                 time vector added in the middle.
    3. SelfAttention:            lets every position look at every other position
                                 ("the ears should match the face"). Used once, at the smallest size.
    4. Downsample / Upsample:    halve / double the image size.
    5. UNet:                     put it all together.

NEW PYTORCH LAYERS YOU WILL MEET
    nn.Conv2d(in, out, 3, padding=1)   looks at 3x3 pixel neighbourhoods; keeps the size
    nn.GroupNorm(8, channels)          keeps the numbers in a healthy range (makes training stable)
    nn.SiLU() / F.silu                 a smooth activation function (like ReLU)
    torch.cat([a, b], dim=1)           joins two feature maps along the channel axis (used for skips)

HOW TO RUN
    python step3_unet.py   (works on a laptop CPU in seconds; no real training happens)

DONE WHEN
    [ ] the output shape == the input shape, e.g. (4, 3, 64, 64) → (4, 3, 64, 64)
    [ ] it prints the parameter count (millions: small compared to Stable Diffusion's ~860 million)
    [ ] the "can it learn?" check shows the loss going down

NEXT: step4_train.py
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BASE_CHANNELS, CHANNEL_MULTS, CHANNELS, IMAGE_SIZE, NUM_CLASSES, TIME_EMB_DIM, USE_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# 1. TIMESTEP EMBEDDING: t (a single number) → a vector the network can use
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalTimeEmbedding(nn.Module):
    """t (B,) → vector (B, dim) made of sines and cosines at many frequencies.

    Why not just feed t = 537 in directly? One raw number is a weak signal, and big values like 537
    would swamp everything else. Sines and cosines at many speeds turn t into a rich pattern:
    the slow waves tell "roughly where in 0..999", the fast waves tell exact small differences.
    Every t gets its own unique fingerprint, and nearby t's get similar fingerprints.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        # `half` frequencies, from fast (1) to very slow (1/10000)
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        angles = t[:, None].float() * frequencies[None, :]            # (B, 1) * (1, half) → (B, half)
        return torch.cat([angles.sin(), angles.cos()], dim=1)        # (B, dim)


# ─────────────────────────────────────────────────────────────────────────────
# 2. RESBLOCK: the main workhorse
# ─────────────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """Two 3x3 convolutions that also receive the time vector, plus a residual ("add the input back").

    x ──► norm → silu → conv1 ──► + time ──► norm → silu → dropout → conv2 ──► + ──► out
    │                                                                         ▲
    └──────────────────────────── skip (1x1 conv if the channels change) ─────┘

    Residual = the block only has to learn a CHANGE to its input, not rebuild it from scratch.
    That makes deep networks much easier to train.
    """

    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)       # squeeze the time vector to match out_ch
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)                 # randomly zero 10% of values while training:
                                                           # stops the model memorising the 20k photos
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        # If the channel count changes, the input must be reshaped before it can be added back
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        # Add the time info to EVERY pixel: (B, out_ch) → (B, out_ch, 1, 1), then broadcast over H and W
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SELF-ATTENTION: every position looks at every other position
# ─────────────────────────────────────────────────────────────────────────────
class SelfAttention(nn.Module):
    """Convolutions only see neighbours. Attention lets each of the 8x8 = 64 positions look at all
    the others, so far-apart parts of the dog stay consistent (two ears, matching eyes).
    It's expensive at big sizes, so we only use it at the smallest size (8x8).
    This is the same attention used in transformers / ChatGPT, applied to image patches."""

    def __init__(self, channels, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)               # (B, C, H, W) → (B, H*W, C): a list of 64 "tokens"
        h, _ = self.attention(h, h, h, need_weights=False)        # each token looks at all tokens
        return x + h.transpose(1, 2).reshape(B, C, H, W)         # back to an image, + residual


# ─────────────────────────────────────────────────────────────────────────────
# 4. DOWNSAMPLE / UPSAMPLE
# ─────────────────────────────────────────────────────────────────────────────
class Downsample(nn.Module):
    """Halve the height and width. stride=2 = the 3x3 window jumps 2 pixels at a time."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Double the height and width: copy each pixel into a 2x2 block, then smooth it with a conv."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ─────────────────────────────────────────────────────────────────────────────
# 5. THE U-NET
# ─────────────────────────────────────────────────────────────────────────────
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ── TIME: t → sinusoidal vector → small MLP → TIME_EMB_DIM numbers ──
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(BASE_CHANNELS),
            nn.Linear(BASE_CHANNELS, TIME_EMB_DIM),
            nn.SiLU(),
            nn.Linear(TIME_EMB_DIM, TIME_EMB_DIM),
        )

        # ── LABEL (STEP 6: "draw THIS breed") ──
        # One learnable vector per breed, plus one extra "no label" vector (id = NUM_CLASSES)
        # that classifier-free guidance needs. It gets ADDED to the time vector.
        if USE_LABELS:
            self.label_emb = nn.Embedding(NUM_CLASSES + 1, TIME_EMB_DIM)

        # ── IN: image channels (1 or 3) → BASE_CHANNELS feature maps ──
        self.in_conv = nn.Conv2d(CHANNELS, BASE_CHANNELS, kernel_size=3, padding=1)

        # ── DOWN PATH ──
        # Per level: 2 ResBlocks, then a Downsample (except at the last level).
        # Every output is saved as a skip connection, so we record each one's channel count.
        self.down = nn.ModuleList()        # ModuleList, not a Python list, or PyTorch won't find the weights!
        skip_channels = [BASE_CHANNELS]    # the in_conv output is the first skip
        ch = BASE_CHANNELS
        last_level = len(CHANNEL_MULTS) - 1
        for level, mult in enumerate(CHANNEL_MULTS):
            out_ch = BASE_CHANNELS * mult
            for _ in range(2):
                self.down.append(ResBlock(ch, out_ch, TIME_EMB_DIM))
                ch = out_ch
                skip_channels.append(ch)
            if level != last_level:
                self.down.append(Downsample(ch))
                skip_channels.append(ch)

        # ── MIDDLE: at the smallest size ──
        self.mid_block1 = ResBlock(ch, ch, TIME_EMB_DIM)
        self.mid_attention = SelfAttention(ch)
        self.mid_block2 = ResBlock(ch, ch, TIME_EMB_DIM)

        # ── UP PATH: the mirror image ──
        # Per level: 3 ResBlocks (one for each skip saved at that level), then an Upsample
        # (except at the top). Each ResBlock's input = current channels + skip channels (torch.cat).
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(CHANNEL_MULTS))):
            out_ch = BASE_CHANNELS * mult
            for _ in range(3):
                self.up.append(ResBlock(ch + skip_channels.pop(), out_ch, TIME_EMB_DIM))
                ch = out_ch
            if level != 0:
                self.up.append(Upsample(ch))

        # ── OUT: feature maps → CHANNELS numbers per pixel = the predicted noise ──
        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, CHANNELS, kernel_size=3, padding=1),
        )

    def forward(self, x, t, labels=None, verbose=False):
        """x: noisy images (B, C, H, W) · t: timesteps (B,) · labels: breed numbers (B,) or None"""
        show = print if verbose else (lambda *args: None)

        t_emb = self.time_mlp(t)                                   # (B, TIME_EMB_DIM)
        if labels is not None:                                     # STEP 6
            t_emb = t_emb + self.label_emb(labels)
        show(f"{'input':<22}{tuple(x.shape)}   time vector {tuple(t_emb.shape)}")

        h = self.in_conv(x)
        skips = [h]
        show(f"{'in_conv':<22}{tuple(h.shape)}")

        for layer in self.down:                                    # DOWN: remember every output
            h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)
            show(f"{'down ' + type(layer).__name__:<22}{tuple(h.shape)}")

        h = self.mid_block1(h, t_emb)                              # MIDDLE
        h = self.mid_attention(h)
        h = self.mid_block2(h, t_emb)
        show(f"{'middle (+attention)':<22}{tuple(h.shape)}")

        for layer in self.up:                                      # UP: join each saved skip back in
            if isinstance(layer, ResBlock):
                h = torch.cat([h, skips.pop()], dim=1)             # stack the skip onto the channels
                h = layer(h, t_emb)
            else:
                h = layer(h)
            show(f"{'up ' + type(layer).__name__:<22}{tuple(h.shape)}")

        out = self.out(h)
        show(f"{'output (noise guess)':<22}{tuple(out.shape)}")
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Tests (no real training)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)

    print("===== 1. TIME EMBEDDING: every t gets its own fingerprint =====")
    embed = SinusoidalTimeEmbedding(BASE_CHANNELS)
    fingerprints = embed(torch.tensor([0, 1, 500, 999]))
    for t, vector in zip([0, 1, 500, 999], fingerprints):
        print(f"t = {t:3d} → first 5 of {len(vector)} numbers: {[round(v, 2) for v in vector[:5].tolist()]}")
    similar = F.cosine_similarity(fingerprints[0], fingerprints[1], dim=0).item()
    different = F.cosine_similarity(fingerprints[0], fingerprints[3], dim=0).item()
    print(f"similarity t=0 vs t=1: {similar:.2f} (close) | t=0 vs t=999: {different:.2f} (far apart)\n")

    print("===== 2. FOLLOW A BATCH THROUGH THE U-NET (watch the U shape) =====")
    model = UNet()
    x = torch.randn(4, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)          # 4 fake noisy images
    t = torch.randint(0, 1000, (4,))                               # a random timestep for each
    start = time.time()
    out = model(x, t, verbose=True)
    print(f"\ninput {tuple(x.shape)} → output {tuple(out.shape)} | same shape: {x.shape == out.shape}"
          f" | one forward pass on {x.device}: {time.time() - start:.2f}s")
    parameters = sum(p.numel() for p in model.parameters())
    print(f"parameters: {parameters:,}  (step 0's classifier had 101,770; Stable Diffusion's U-Net ~860,000,000)\n")

    print("===== 3. CAN IT LEARN? (a quick sanity check, NOT real training) =====")
    # A real denoising exercise: 4 real images, noised with STEP 2's add_noise, and the U-Net must
    # find the noise hidden inside them. We repeat the SAME exercise 30 times. If the loss doesn't
    # drop, something is broken (e.g. gradients not flowing). Real training on varied data is STEP 4.
    # (Guessing "0 everywhere" gives a loss of about 1.0, so we want to go well below that.)
    from step1_dataset import get_dataset
    from step2_noise_schedule import add_noise, make_schedule

    dataset = get_dataset()
    x0 = torch.stack([dataset[i][0] for i in [0, len(dataset) // 4, len(dataset) // 2, 3 * len(dataset) // 4]])
    t = torch.tensor([50, 300, 600, 900])                         # four different noise levels
    noise = torch.randn_like(x0)
    x_t = add_noise(x0, t, noise, make_schedule(device="cpu"))

    model = UNet()                                                 # a fresh, untrained U-Net
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for i in range(30):
        loss = F.mse_loss(model(x_t, t), noise)                    # how far is the guess from the real noise?
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 5 == 0 or i == 29:
            print(f"  attempt {i + 1:2d}: loss {loss.item():.4f}")
    print("the loss went well below 1.0 → gradients flow through every layer; the U-Net is ready for STEP 4")
    print("(takes ~1 minute on a laptop CPU for dogs, a few seconds for MNIST)")
