"""
STEP 1 of 7: THE DATASET
========================
GOAL
    Give the training loop batches of clean images, scaled to -1..1, shaped
    (batch, CHANNELS, IMAGE_SIZE, IMAGE_SIZE), plus a label for each image.

WHY THIS MATTERS
    The model can only learn what is in the data. Clean, well-cropped images
    give much better results than a bigger, messier model.

IMAGES ARE JUST NUMBERS
    A photo file:      (height, width, 3) whole numbers 0..255
    What PyTorch uses: (channels, height, width) floats, and we scale them to -1..1
    Why -1..1? Diffusion adds noise that is centred on 0 (mostly between -1 and 1),
    so the images are scaled to the same range as the noise.
    torchvision's transforms do this conversion for us:
        transforms.ToTensor()                    → (C, H, W), values 0..1
        transforms.Normalize(mean=0.5, std=0.5)  → (x - 0.5) / 0.5  → values -1..1

TWO DATASETS (switch with DATASET in config.py)
    "mnist": 60,000 handwritten digits, 28x28 grayscale. torchvision downloads it.
             We pad it to 32x32 so the U-Net can halve the size cleanly (32 → 16 → 8).
             This is the PRACTICE ROUND: it trains in minutes, so bugs show up fast.

    "dogs":  Stanford Dogs, 20,580 photos of 120 breeds, with a box around each dog.
             We use the Hugging Face copy (the original Stanford server is often offline):
               https://huggingface.co/datasets/Alanox/stanford-dogs
               images.tar.gz → extract to data/stanford_dogs/images/   (one folder of .jpg files)
               metadata.csv  → data/stanford_dogs/metadata.csv
             (train_on_colab.ipynb downloads both for you on Colab)
             metadata.csv has one row per photo:
               name                annotations              target
               n02093647_1538.jpg  "[[0, 22, 476, 476]]"    Bedlington Terrier
               └ the file          └ box(es) around the dog └ the breed
             Each box is [xmin, ymin, xmax, ymax]. Some photos have several dogs (several boxes).
             Why crop to the box? In many photos the dog is small; cropping makes the dog fill the image.

HOW TO RUN
    python step1_dataset.py   → saves samples/step1_real_images.png (a 4x4 grid of training images)

DONE WHEN
    [ ] the grid shows correctly cropped training images
    [ ] one batch has shape (BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE) and values between -1 and 1

NEXT: step2_noise_schedule.py
"""
import ast
import csv

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from config import BATCH_SIZE, DATA_DIR, DATASET, DEVICE, IMAGE_SIZE, SAMPLES_DIR

DOGS_DIR = DATA_DIR / "stanford_dogs"
MNIST_CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


# ─────────────────────────────────────────────────────────────────────────────
# MILESTONE 1: MNIST (torchvision does most of the work)
# ─────────────────────────────────────────────────────────────────────────────
def get_mnist_dataset():
    transform = transforms.Compose([      # Compose = apply these one after another, to every image
        transforms.Pad(2),                # 28x28 → 32x32: adds a 2-pixel black border on every side
        transforms.ToTensor(),            # picture → tensor (1, 32, 32), values 0..1
        transforms.Normalize([0.5], [0.5]),   # (x - 0.5) / 0.5 → values -1..1
    ])
    # The test set is not needed: a generator has no "right answer" to test against.
    return datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)


# ─────────────────────────────────────────────────────────────────────────────
# MILESTONE 2: Stanford Dogs (we write our own Dataset class)
# ─────────────────────────────────────────────────────────────────────────────
def biggest_box(boxes):
    """Some photos have 2–6 dogs. Pick the biggest one: it's the main subject and has the most detail."""
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))    # area = width × height


class StanfordDogs(Dataset):
    """A PyTorch Dataset needs only three methods:
        __init__     → find all the files (runs once)
        __len__      → how many images there are
        __getitem__  → load and prepare image number `index` (runs once per image, every epoch)
    """

    def __init__(self, root=DOGS_DIR, augment=True):
        images_dir, metadata_path = root / "images", root / "metadata.csv"
        if not metadata_path.exists() or not images_dir.exists():
            raise FileNotFoundError(
                f"Stanford Dogs not found in {root}. Download images.tar.gz and metadata.csv from "
                "https://huggingface.co/datasets/Alanox/stanford-dogs and extract the images "
                f"into {images_dir} (or run train_on_colab.ipynb on Colab).")

        # Read the CSV: one row per photo → {"name": ..., "annotations": ..., "target": ...}
        with open(metadata_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        # Give every breed a number. Sorting keeps the numbers stable: the same breed always
        # gets the same number, on every computer and every run.
        # "Golden Retriever" → "golden retriever"   (STEP 6 uses these names as the "text")
        self.class_names = sorted({row["target"].lower() for row in rows})
        label_of = {name: number for number, name in enumerate(self.class_names)}

        # Every photo becomes one entry: (image file, the box around the dog, its breed number)
        self.samples = []
        for row in rows:
            boxes = ast.literal_eval(row["annotations"])   # the text "[[0, 22, 476, 476]]" → a real Python list
            self.samples.append((images_dir / row["name"], biggest_box(boxes), label_of[row["target"].lower()]))

        steps = [
            transforms.Resize(IMAGE_SIZE),          # shorter side → IMAGE_SIZE (keeps the shape, no squashing)
            transforms.CenterCrop(IMAGE_SIZE),      # cut the middle square → IMAGE_SIZE x IMAGE_SIZE
        ]
        if augment:
            steps.append(transforms.RandomHorizontalFlip())   # 50% chance to mirror: a mirrored dog is still
                                                              # a dog, so this is free extra training data
        steps += [
            transforms.ToTensor(),                            # → (3, H, W), values 0..1
            transforms.Normalize([0.5] * 3, [0.5] * 3),       # → values -1..1 (one mean/std per colour channel)
        ]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, (x1, y1, x2, y2), label = self.samples[index]
        img = Image.open(image_path).convert("RGB")   # "RGB": a few photos are grayscale or CMYK; make them all the same

        # Crop to the dog. A few boxes go slightly outside the photo, so keep them inside it.
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        img = img.crop((x1, y1, x2, y2))

        return self.transform(img), label


# ─────────────────────────────────────────────────────────────────────────────
# The functions the rest of the project uses
# ─────────────────────────────────────────────────────────────────────────────
def get_dataset():
    return get_mnist_dataset() if DATASET == "mnist" else StanfordDogs()


def get_class_names(dataset):
    """The name of every label number. STEP 6 needs these to turn text into a label."""
    return MNIST_CLASS_NAMES if DATASET == "mnist" else dataset.class_names


def get_dataloader(dataset=None):
    dataset = dataset or get_dataset()
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,                    # a new random order every epoch
        num_workers=2,                   # load the next batches in 2 background processes while the GPU trains
        drop_last=True,                  # skip the last, smaller batch so every batch has the same size
        pin_memory=(DEVICE == "cuda"),   # makes copying batches to the GPU a little faster
    )


if __name__ == "__main__":
    dataset = get_dataset()
    class_names = get_class_names(dataset)
    print(f"dataset: {DATASET} | {len(dataset):,} images | {len(class_names)} classes")

    images, labels = next(iter(get_dataloader(dataset)))
    print("batch of images:", tuple(images.shape), "→ (batch, channels, height, width)")
    print(f"pixel values: min {images.min().item():.2f}, max {images.max().item():.2f}  (should be -1 and 1)")
    print("first 16 labels:", [class_names[i] for i in labels[:16].tolist()])

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    # save_image puts a batch into one grid picture. normalize + value_range tell it our pixels are -1..1.
    save_image(images[:16], SAMPLES_DIR / "step1_real_images.png", nrow=4, normalize=True, value_range=(-1, 1))
    print("saved", SAMPLES_DIR / "step1_real_images.png")
