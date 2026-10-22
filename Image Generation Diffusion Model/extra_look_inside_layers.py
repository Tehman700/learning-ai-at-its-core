"""
EXTRA: LOOK INSIDE THE LAYERS (what does each layer "see" in one dog photo?)
=============================================================================
Not part of the 7 steps. A tool for UNDERSTANDING: run it now, and again after STEP 4.

PART 1: A TRAINED network (ResNet-18, already trained on ImageNet by PyTorch)
    Our U-Net is untrained, so its layers would only show random patterns. ResNet-18 is built from
    the same ingredients as our U-Net's DOWN path (convolutions, ResBlocks, shrinking step by step),
    and it is fully trained, so you can see what layers REALLY learn:
        early layers → edges and colours,  middle layers → textures and parts,  deep layers → "dog".
    It also says which breed it thinks the dog is, and WHERE in the photo it looked to decide.
    (The ResNet-18 weights, ~45 MB, download automatically the first time.)

PART 2: OUR U-Net on the same dog (needs DATASET = "dogs" in config.py)
    The dog is noised to t = 300 (STEP 2) and passed through our U-Net (STEP 3).
    Untrained: the maps are random mixes of the input, and the noise guess is useless.
    After STEP 4 (checkpoints/dogs.pt exists), the SAME script loads the trained weights and
    the predicted noise starts to look like the real noise.

HOW TO RUN
    python extra_look_inside_layers.py
    → samples/inside_resnet.png   (trained network, layer by layer + where it looked)
    → samples/inside_unet.png     (our U-Net, layer by layer + its noise guess)

HOW TO READ THE PICTURES
    Each small square is ONE feature map ("pattern detector"). Bright = "my pattern is HERE".
    We show the 8 most active maps of each layer (a layer has 64 to 512 of them).
"""
import ast
import csv

import matplotlib
matplotlib.use("Agg")                     # draw to files, no window needed
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import DATA_DIR, DATASET, SAMPLES_DIR

DOG = "n02099601_4005.jpg"                # a golden retriever; try any file name from metadata.csv
MAPS_PER_LAYER = 8


def load_dog(size):
    """Load the dog photo, crop to its box, make it size x size. Returns (PIL image, breed name)."""
    with open(DATA_DIR / "stanford_dogs" / "metadata.csv", encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["name"] == DOG)
    boxes = ast.literal_eval(row["annotations"])
    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    img = Image.open(DATA_DIR / "stanford_dogs" / "images" / DOG).convert("RGB").crop(box)
    return transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size)])(img), row["target"]


def most_active(feature_maps, n=MAPS_PER_LAYER):
    """feature_maps (C, H, W) → the n maps with the highest average activation."""
    strength = feature_maps.mean(dim=(1, 2))
    return feature_maps[strength.argsort(descending=True)[:n]]


def plot_rows(rows, first_column, path, title):
    """rows: list of (label, maps (n, H, W)). first_column: an image shown at the start of every row."""
    fig, axes = plt.subplots(len(rows), MAPS_PER_LAYER + 1, figsize=(1.6 * (MAPS_PER_LAYER + 1), 1.75 * len(rows)))
    for r, (label, maps) in enumerate(rows):
        axes[r, 0].imshow(first_column)
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
        for c in range(MAPS_PER_LAYER):
            if c < len(maps):
                axes[r, c + 1].imshow(maps[c], cmap="magma")
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: a trained network
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def part1_trained_resnet():
    from torchvision.models import ResNet18_Weights, resnet18

    print("===== PART 1: A TRAINED NETWORK (ResNet-18) LOOKS AT THE DOG =====")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval()

    img, breed = load_dog(224)
    x = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),   # the scaling ResNet was trained with
    ])(img)[None]                                                             # (1, 3, 224, 224)

    # "Hooks" let us grab each layer's output while the image flows through
    layers = [
        ("relu",   "first conv\n64 maps, 112×112\nedges & colours"),
        ("layer1", "ResBlocks 1\n64 maps, 56×56\ntextures (fur)"),
        ("layer2", "ResBlocks 2\n128 maps, 28×28\nsmall parts"),
        ("layer3", "ResBlocks 3\n256 maps, 14×14\neyes, nose, ears"),
        ("layer4", "ResBlocks 4\n512 maps, 7×7\n'dog' concepts"),
    ]
    captured = {}
    for name, _ in layers:
        getattr(model, name).register_forward_hook(lambda m, i, out, name=name: captured.__setitem__(name, out[0]))

    probabilities = model(x).softmax(dim=1)[0]

    rows = [(label, most_active(captured[name])) for name, label in layers]

    # WHERE did it look? (a "class activation map": the last layer's maps, weighted by how much
    # each one votes for the predicted breed)
    top = probabilities.argmax().item()
    cam = (model.fc.weight[top][:, None, None] * captured["layer4"]).sum(dim=0)     # (7, 7)
    cam = F.interpolate(cam[None, None], size=224, mode="bilinear")[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    plot_rows(rows, img, SAMPLES_DIR / "inside_resnet.png",
              f"A TRAINED network looking at a {breed}: each row = one layer, each square = one pattern detector")

    # The answer + where it looked, as a separate small picture
    names = weights.meta["categories"]
    top5 = probabilities.topk(5)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img); axes[0].set_title(f"the photo (true breed: {breed})", fontsize=9)
    axes[1].imshow(img); axes[1].imshow(cam, cmap="jet", alpha=0.45)
    axes[1].set_title("where the network looked (red = most important)", fontsize=9)
    for ax in axes:
        ax.axis("off")
    guesses = "   ".join(f"{names[i]} {p:.0%}" for p, i in zip(top5.values.tolist(), top5.indices.tolist()))
    fig.suptitle("network's top guesses: " + guesses, fontsize=9)
    fig.savefig(SAMPLES_DIR / "inside_resnet_decision.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", SAMPLES_DIR / "inside_resnet_decision.png")
    print("true breed:", breed)
    for p, i in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  guess: {names[i]:<25} {p:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: our U-Net
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def part2_our_unet():
    print("\n===== PART 2: OUR U-NET LOOKS AT THE SAME DOG (noised to t = 300) =====")
    if DATASET != "dogs":
        print('skipped: set DATASET = "dogs" in config.py to run this part')
        return

    from step2_noise_schedule import add_noise, make_schedule
    from step3_unet import ResBlock, UNet

    model = UNet().eval()
    from step4_train import CHECKPOINT_PATH as checkpoint
    trained = checkpoint.exists()
    if trained:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["ema"])
    status = f"TRAINED (loaded {checkpoint.name})" if trained else "UNTRAINED (random weights: run again after STEP 4!)"
    print("U-Net is", status)

    img, breed = load_dog(64)
    x0 = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])(img)[None]
    torch.manual_seed(0)
    noise = torch.randn_like(x0)
    t = torch.tensor([300])
    x_t = add_noise(x0, t, noise, make_schedule(device="cpu"))

    # Grab the output of the LAST ResBlock at each level (just before the size changes)
    def level_ends(blocks):
        ends = []
        for i, block in enumerate(blocks):
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            if isinstance(block, ResBlock) and not isinstance(nxt, ResBlock):
                ends.append(block)
        return ends

    watch = [(model.in_conv, "in_conv")]
    watch += [(b, f"down level {i + 1}") for i, b in enumerate(level_ends(model.down))]
    watch += [(model.mid_block2, "bottom\n(+ attention)")]
    watch += [(b, f"up level {i + 1}") for i, b in enumerate(level_ends(model.up))]
    captured = []
    for module, label in watch:
        module.register_forward_hook(lambda m, i, out, label=label: captured.append((label, out[0])))

    predicted_noise = model(x_t, t)[0]

    rows = []
    for label, maps in captured:
        rows.append((f"{label}\n{maps.shape[0]} maps, {maps.shape[1]}×{maps.shape[2]}", most_active(maps)))

    noisy_picture = ((x_t[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
    plot_rows(rows, noisy_picture, SAMPLES_DIR / "inside_unet.png",
              f"OUR U-Net ({status.split(' (')[0]}) looking at a noisy {breed} (t = 300)")

    # The final answer: predicted noise vs the real noise
    error = F.mse_loss(predicted_noise, noise[0]).item()
    show = lambda z: ((z.clamp(-2.5, 2.5) + 2.5) / 5).permute(1, 2, 0).numpy()
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for ax, picture, title in zip(axes,
                                  [((x0[0] + 1) / 2).permute(1, 2, 0).numpy(), noisy_picture, show(noise[0]), show(predicted_noise)],
                                  ["clean dog x0", "noisy input x_t (t=300)", "REAL noise ε", f"U-Net's GUESS ε̂\n(MSE {error:.3f})"]):
        ax.imshow(picture); ax.set_title(title, fontsize=9); ax.axis("off")
    fig.savefig(SAMPLES_DIR / "inside_unet_answer.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", SAMPLES_DIR / "inside_unet_answer.png")
    print(f"noise-guess error (MSE): {error:.3f}   (untrained ≈ 1 or more; well trained ≈ 0.05–0.1 at t=300)")


if __name__ == "__main__":
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    part1_trained_resnet()
    part2_our_unet()
