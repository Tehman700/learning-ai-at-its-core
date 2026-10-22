"""
STEP 0 of 7: PYTORCH BASICS
===========================
GOAL
    Train a tiny neural network that recognises handwritten digits (MNIST).
    It has nothing to do with diffusion yet. It teaches you the training loop, and
    STEP 4 (training the diffusion model) is the SAME loop with a different loss.

WHY THIS MATTERS
    Every neural network, from this toy one to Stable Diffusion, is trained like this:
        1. forward:  give the model an input and get its prediction
        2. loss:     one number that says how wrong the prediction is
        3. backward: PyTorch works out how to change every weight to reduce the loss
        4. step:     the optimizer changes the weights a little
        5. repeat thousands of times

WHAT YOU WILL LEARN
    - tensors: PyTorch's arrays (like numpy) that can live on a GPU
    - autograd: PyTorch computes gradients (the "backward" step) for you
    - nn.Module: how you define a network
    - Dataset / DataLoader: how you feed data in batches

HOW TO RUN
    python step0_pytorch_basics.py
    (MNIST downloads automatically the first time, ~12 MB, into data/)
    On a laptop CPU the whole file takes a few minutes (including the first download).

DONE WHEN
    [ ] part 1 prints tensor shapes and where the tensor lives (cpu or cuda)
    [ ] part 2 prints the gradient 6.0
    [ ] part 5 prints a loss that goes DOWN
    [ ] part 6 prints a test accuracy above ~95%

NEXT: step1_dataset.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import DATA_DIR, DEVICE


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Tensors
# A tensor is a grid of numbers: 0-D (one number), 1-D (a list), 2-D (a table),
# 3-D, 4-D... Images are 4-D tensors: (batch, channels, height, width).
# ─────────────────────────────────────────────────────────────────────────────
def part1_tensors():
    print("\n===== PART 1: TENSORS =====")

    # A 2x2 tensor made from a Python list. The "." makes the numbers floats (decimals).
    a = torch.tensor([[1., 2.], [3., 4.]])
    print("a =\n", a)
    print("shape:", a.shape, "| dtype:", a.dtype, "| device:", a.device)

    # A fake batch of 8 grayscale 28x28 images filled with random numbers.
    # PyTorch ALWAYS orders image dimensions as (batch, channels, height, width).
    images = torch.randn(8, 1, 28, 28)
    print("fake image batch shape:", images.shape)

    # Maths works on every element at once, with no Python loops needed.
    print("a + 10 =\n", a + 10)
    print("a * a  =\n", a * a)            # element by element
    print("a @ a  =\n", a @ a)            # matrix multiplication (what Linear layers do inside)
    print("mean of a:", a.mean().item())  # .item() turns a 1-number tensor into a normal Python float

    # Tensors can move to the GPU. On Colab with a T4 this prints "cuda:0"; on a laptop, "cpu".
    a = a.to(DEVICE)
    print("a now lives on:", a.device)


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Autograd (automatic gradients)
# A gradient answers: "if I nudge this number up a little, does the loss go up or down,
# and by how much?" Training is just following gradients downhill.
# ─────────────────────────────────────────────────────────────────────────────
def part2_autograd():
    print("\n===== PART 2: AUTOGRAD =====")

    x = torch.tensor(3.0, requires_grad=True)  # requires_grad=True → "keep track of what happens to x"
    y = x ** 2                                 # y = x²
    y.backward()                               # compute dy/dx and store it in x.grad
    print("y = x² at x = 3 → dy/dx =", x.grad.item(), "(the maths says 2x = 6)")

    # In a network PyTorch does exactly this, but for millions of weights at once.


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: A network (nn.Module)
# Every PyTorch model is a class that inherits nn.Module and has:
#   __init__  → create the layers (they hold the learnable weights)
#   forward   → say how data flows through the layers
# ─────────────────────────────────────────────────────────────────────────────
class TinyClassifier(nn.Module):
    """28x28 image → 10 scores (one per digit). The highest score is the prediction."""

    def __init__(self):
        super().__init__()                # always call this first in an nn.Module
        self.layers = nn.Sequential(      # Sequential = run these layers one after another
            nn.Flatten(),                 # (B, 1, 28, 28) → (B, 784): a flat list of pixels
            nn.Linear(784, 128),          # 784 inputs → 128 outputs (784*128 + 128 = 100,480 weights)
            nn.ReLU(),                    # keep positives, turn negatives to 0. Without a non-linearity,
                                          # stacked Linear layers collapse into ONE linear layer.
            nn.Linear(128, 10),           # 128 → 10 scores, one per digit 0..9
        )

    def forward(self, x):
        return self.layers(x)


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: Data
# Dataset    = the collection of (image, label) pairs
# DataLoader = serves the Dataset in shuffled batches
# ─────────────────────────────────────────────────────────────────────────────
def part4_get_data():
    print("\n===== PART 4: DATA =====")

    transform = transforms.ToTensor()   # picture (28x28, values 0..255) → tensor (1, 28, 28), values 0..1

    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)   # 60,000 images
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)   # 10,000 images
    # The test set is NEVER used for training. It checks whether the model learned digits in general
    # or just memorised the training images.

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)    # shuffle: a new order every epoch
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)    # order doesn't matter for testing

    images, labels = next(iter(train_loader))   # grab one batch to look at
    print("one batch of images:", images.shape)  # (64, 1, 28, 28)
    print("one batch of labels:", labels.shape, "→ first 10:", labels[:10].tolist())
    print("pixel values go from", images.min().item(), "to", images.max().item())

    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# PART 5: The training loop ← the most important part of this file
# ─────────────────────────────────────────────────────────────────────────────
def part5_train(model, train_loader, epochs=2):
    print("\n===== PART 5: TRAINING =====")

    loss_fn = nn.CrossEntropyLoss()      # the standard loss for "pick one of N classes"
                                         # (STEP 4 uses MSE instead: "how far is the predicted noise from the real noise")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # lr = learning rate = how big each update step is

    model.train()                        # training mode (matters for some layer types; a good habit)
    for epoch in range(epochs):          # 1 epoch = the model has seen every training image once
        running_loss = 0.0
        for batch_number, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)   # the data must be on the same device as the model

            predictions = model(images)              # 1. FORWARD:  (64, 10) scores
            loss = loss_fn(predictions, labels)      # 2. LOSS:     one number, how wrong we are

            optimizer.zero_grad()                    #    clear the old gradients (otherwise they add up)
            loss.backward()                          # 3. BACKWARD: a gradient for every weight
            optimizer.step()                         # 4. STEP:     nudge every weight downhill

            running_loss += loss.item()
            if (batch_number + 1) % 200 == 0:
                print(f"epoch {epoch + 1} | batch {batch_number + 1:4d}/{len(train_loader)} "
                      f"| average loss {running_loss / 200:.4f}")
                running_loss = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PART 6: Evaluate on images the model has NEVER seen
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()   # no gradients needed when only testing: faster, uses less memory
def part6_evaluate(model, test_loader):
    print("\n===== PART 6: EVALUATION =====")

    model.eval()                         # evaluation mode (the partner of model.train())
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        predicted_digit = model(images).argmax(dim=1)     # the position of the highest score = the predicted digit
        correct += (predicted_digit == labels).sum().item()
        total += labels.shape[0]
    print(f"test accuracy: {correct}/{total} = {100 * correct / total:.2f}%")

    # Show a few individual predictions
    images, labels = next(iter(test_loader))
    predicted = model(images.to(DEVICE)).argmax(dim=1).cpu()
    print("real digits:     ", labels[:10].tolist())
    print("model predicted: ", predicted[:10].tolist())


if __name__ == "__main__":
    torch.manual_seed(0)                 # fixes the randomness, so you get the same results every run

    part1_tensors()
    part2_autograd()

    train_loader, test_loader = part4_get_data()
    model = TinyClassifier().to(DEVICE)  # move all the weights to the GPU (if there is one)
    print("\nthe model has", sum(p.numel() for p in model.parameters()), "learnable numbers (parameters)")

    part5_train(model, train_loader)
    part6_evaluate(model, test_loader)
