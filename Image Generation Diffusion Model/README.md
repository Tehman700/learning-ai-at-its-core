# Image Generation with a Diffusion Model, from Scratch

**Goal:** build a diffusion model in PyTorch that generates dog images.
You type a breed (e.g. `golden retriever`) and it draws one, starting from pure noise.

Training runs on **Google Colab** (free T4 GPU). Small tests run on your laptop.

---

## The whole process in one picture

```
 TRAINING                                                   GENERATING
 ────────                                                   ──────────
 dog photo ──crop, resize────────► clean image x0         breed name ──► label ID
                                          │                                  │
 pick a random step t (0..999)            │                 pure noise x_T   │
 make random noise ε                      │                       │          │
                                          ▼                       ▼          ▼
            x_t = mix of x0 and ε  (more noise when t is big)   ┌──────────────────┐
                                          │                     │  U-Net predicts   │
                                          ▼                     │  the noise, we    │
            ┌──────────── U-Net ────────────┐                   │  remove a little  │
            │ input: x_t, t, (breed label)  │                   │  of it            │
            │ output: predicted noise ε̂     │                   └──────────────────┘
            └───────────────────────────────┘                     repeat 1000 times
                                          │                       (or ~50 with DDIM)
            loss = mean((ε - ε̂)²)  ──► backprop ──► update          │
            repeat for many thousands of batches                     ▼
                                                                   dog image 🐕
```

**The idea in one sentence:** we never teach the network to *draw*. We teach it to *remove noise*. If it can remove noise really well, then starting from pure noise and removing noise step by step produces a brand-new image.

---

## The steps (do them in this order)

Each file starts with a comment block that explains the goal and the theory, followed by the code with comments on every important line.
Read it, run the file (its test is at the bottom), check the "done when" list, then move to the next step.

| Step | File | What you build | You learn |
|---|---|---|---|
| 0 | `step0_pytorch_basics.py` | A tiny digit classifier | Tensors, autograd, `nn.Module`, the training loop |
| 1 | `step1_dataset.py` | MNIST and Stanford Dogs datasets | Images as numbers, Datasets, DataLoaders, bounding-box cropping |
| 2 | `step2_noise_schedule.py` | The forward process (adding noise) | The diffusion maths: β, α, ᾱ |
| 3 | `step3_unet.py` | The U-Net denoiser | Convolutions, ResBlocks, skip connections, timestep embedding |
| 4 | `step4_train.py` | The training loop | Noise-prediction loss, AMP, EMA, checkpoints |
| 5 | `step5_sample.py` | The reverse process (generating) | DDPM sampling; a GIF of noise turning into an image |
| 6 | `step6_text_conditioning.py` | Breed name → image | Conditioning, classifier-free guidance |
| 7 | `step7_app.py` | A web app: type a breed, get a dog | Packaging a model; Gradio |

`config.py` holds every setting (image size, learning rate and so on). Every step imports from it.
Images are handled with `torchvision` (part of PyTorch), so there's no OpenCV dependency.

### Milestones
1. **Milestone 1: digits.** Do steps 0–5 with `DATASET = "mnist"` in `config.py`. The model draws random handwritten digits. This takes minutes to train, so mistakes are cheap.
2. **Milestone 2: dogs.** Set `DATASET = "dogs"` and train again on Colab. The model draws random dogs at 64×64.
3. **Milestone 3: dogs by breed.** Do step 6. Now *you* choose the breed.
4. **Milestone 4: the app.** Do step 7.
5. *(Optional)* Upgrade from breed labels to free text with a CLIP text encoder and cross-attention, then try latent diffusion for bigger images.

---

## Where things run

| Task | Laptop | Colab GPU |
|---|---|---|
| Steps 0–3 (writing and testing pieces) | ✅ | ✅ |
| Training on MNIST (step 4) | Slow, but possible | ✅ ~15–30 min |
| Training on dogs (step 4) | ❌ too slow | ✅ several hours |
| Generating and the app (steps 5–7) | ✅ (slow on CPU) | ✅ |

**Colab workflow:** push this repo to GitHub. Then open `train_on_colab.ipynb` in Colab; it clones the repo, installs packages and runs training. Checkpoints are saved to Google Drive, so a disconnect doesn't lose your progress.

---

## Setup (laptop)
```
pip install -r requirements.txt
```
(Colab already has PyTorch; the notebook installs the rest.)

## Folder layout
```
Image Generation Diffusion Model/
├── README.md                    ← you are here: the roadmap
├── config.py                    ← all settings
├── requirements.txt
├── step0_pytorch_basics.py … step7_app.py   ← the lessons, in order
├── train_on_colab.ipynb         ← runs training on a Colab GPU
├── data/          (created later, not pushed to GitHub: datasets)
├── checkpoints/   (created later, not pushed to GitHub: model weights)
└── samples/       (created later: generated images and videos)
```

## Key words
| Word | Meaning |
|---|---|
| **x0** | A clean, real image |
| **t** | Timestep: how far into the noising process we are (0 = clean, 999 = pure noise) |
| **x_t** | The image after t steps of noise |
| **ε (epsilon)** | The random noise we added |
| **β (beta)** | How much noise is added at one step |
| **ᾱ (alpha bar)** | How much of the original image is left at step t |
| **U-Net** | The neural network that looks at x_t and predicts ε |
| **DDPM / DDIM** | Two ways to run the generation loop (DDIM is faster) |
| **CFG** | Classifier-free guidance: makes the image follow the label more strongly |
| **Checkpoint** | A saved copy of the model's weights |
