"""
STEP 6 of 7: TEXT → IMAGE ("draw a golden retriever")
======================================================
GOAL
    Let the user choose WHAT to draw. So far the model draws a random digit or dog.
    Now we give it the breed as an extra input, a "condition".

HOW CONDITIONING WORKS
    1. Text → label ID:   "golden retriever" → 52   (this file)
    2. Label ID → vector: nn.Embedding(NUM_CLASSES + 1, TIME_EMB_DIM)   (STEP 3)
    3. That vector is ADDED to the time vector, so every ResBlock knows both
       "how noisy is it" and "what should it be".

CLASSIFIER-FREE GUIDANCE (CFG): the trick that makes conditioning actually work
    Training:   10% of the time, hide the label (use the extra "no label" ID = NUM_CLASSES).
                The SAME model learns both "draw a pug" and "draw any dog".
    Generating: ask the model twice and exaggerate the difference:
                    noise = uncond + GUIDANCE_SCALE * (cond - uncond)
                The bigger GUIDANCE_SCALE, the more "pug-like" (and less varied) the images.
    Stable Diffusion uses exactly this trick (its "CFG scale" slider, usually ~7).

CHECKLIST: the changes to make in the OTHER files (search for "STEP 6" in each)
    [ ] config.py:                 USE_LABELS = True
    [ ] step3_unet.py:             add self.label_emb and add it to t_emb in forward()
    [ ] step4_train.py:            pass labels to the model, hiding them CFG_DROP_PROB of the time
    [ ] step4_train.py:            also save the class names in the checkpoint (the app needs them)
    [ ] step5_sample.py:           predict_noise with guidance
    [ ] RETRAIN (the old checkpoint doesn't know about labels): MNIST first, then dogs on Colab
    [ ] this file:                 text_to_label, then test it

HOW TO RUN
    python step6_text_conditioning.py   → saves samples/step6_<label>.png for a few prompts

DONE WHEN
    [ ] MNIST: asking for "7" gives a grid of sevens
    [ ] dogs: "pug" and "golden retriever" give clearly different-looking dogs
    [ ] try GUIDANCE_SCALE = 1, 4 and 8, and describe what changes

OPTIONAL UPGRADE (after the app works): REAL FREE TEXT
    Labels only understand the 120 breed names. To understand ANY sentence ("a fluffy dog on grass"),
    swap step 1+2 for a pretrained CLIP text encoder (it turns a sentence into vectors) and add
    CROSS-ATTENTION layers to the U-Net so each pixel can "look at" the words. That's how Stable
    Diffusion reads prompts. But your captions would also have to describe the images, so it's a bigger project.

NEXT: step7_app.py
"""
import difflib

from config import DATASET

MNIST_CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def normalise(text):
    """'Golden_Retriever ' → 'golden retriever'"""
    # TODO: lowercase it, replace "_" and "-" with spaces, strip the spaces at both ends
    raise NotImplementedError("STEP 6")


def text_to_label(prompt, class_names):
    """
    Turn what the user typed into a label number.
        "golden retriever"                → exact match
        "a photo of a golden retriever"   → a class name appears INSIDE the prompt
        "golden retreiver" (typo)         → closest match (difflib)
    Return None if nothing matches, so the app can say "unknown breed".
    """
    # TODO 1: names = [normalise(n) for n in class_names];  p = normalise(prompt)
    # TODO 2: exact match?        → return its index
    # TODO 3: name inside prompt? → return the index of the LONGEST name found
    #         (so "toy poodle" beats "poodle")
    # TODO 4: typo? difflib.get_close_matches(p, names, n=1, cutoff=0.6) → return its index
    # TODO 5: return None
    raise NotImplementedError("STEP 6")


if __name__ == "__main__":
    # TODO 1: test text_to_label on: "7", "seven"(?), "golden retriever", "a photo of a pug", "pugg", "cat"
    #         ("seven" won't match unless you add word names; try adding them!)
    # TODO 2: generate a grid for a few prompts and save each one:
    #           label  = text_to_label(prompt, class_names)
    #           labels = torch.full((NUM_SAMPLES,), label, device=DEVICE)
    #           images = ddpm_sample(model, schedule, labels=labels)      (from step6)
    pass
