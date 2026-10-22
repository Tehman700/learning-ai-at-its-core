"""
STEP 7 of 7: THE APP (type a breed, get a dog)
==============================================
GOAL
    Wrap everything in one function, generate(prompt), and put a simple web page on it with Gradio.

WHAT HAPPENS WHEN THE USER CLICKS "GENERATE"
    prompt ─► text_to_label (STEP 6) ─► label ID
    random noise ─► sampling loop with guidance (STEP 5) ─► images in -1..1
    ─► converted to pictures with to_pil_image ─► shown on the page

THINGS A REAL APP NEEDS (beginners often forget these)
    - Load the model ONCE when the app starts, not on every click (loading is slow).
    - Use the fast DDIM sampler, or every click waits ~1000 model calls.
    - Handle an unknown breed nicely: show a message, don't crash.

HOW TO RUN
    pip install gradio
    python step7_app.py
    Laptop: open the local link it prints. Colab: use launch(share=True) and open the public link.

DONE WHEN
    [ ] typing "pug" shows a grid of pugs in the browser
    [ ] typing "banana" shows a friendly "unknown breed" message
    [ ] a guidance slider changes how strongly the images follow the breed

YOU BUILT A TEXT-TO-IMAGE DIFFUSION MODEL FROM SCRATCH. 🎉
"""
import gradio as gr
import torch

from config import DEVICE, GUIDANCE_SCALE
from torchvision.transforms.functional import to_pil_image

from step2_noise_schedule import make_schedule
from step5_sample import ddim_sample, load_model
from step6_text_conditioning import text_to_label

# TODO 1: load the model, schedule and class names ONCE, here at the top level
#         (the class names come from the checkpoint; see the STEP 6 checklist)


def generate(prompt, guidance, how_many):
    # TODO 1: label = text_to_label(prompt, class_names)
    #         if label is None: raise gr.Error(f"Unknown breed: {prompt}")
    # TODO 2: labels = torch.full((how_many,), label, device=DEVICE)
    # TODO 3: images = ddim_sample(model, schedule, n=how_many, labels=labels)
    #         (guidance: make ddim_sample / predict_noise accept a guidance value instead of
    #          always using GUIDANCE_SCALE)
    # TODO 4: return [to_pil_image((img.clamp(-1, 1) + 1) / 2) for img in images.cpu()]
    #         (-1..1 → 0..1 → a picture; Gradio shows a list of pictures as a gallery)
    raise NotImplementedError("STEP 7")


if __name__ == "__main__":
    # TODO: app = gr.Interface(
    #           fn=generate,
    #           inputs=[gr.Textbox(label="Dog breed", value="golden retriever"),
    #                   gr.Slider(1, 10, value=GUIDANCE_SCALE, label="Guidance"),
    #                   gr.Slider(1, 16, value=4, step=1, label="How many")],
    #           outputs=gr.Gallery(label="Generated dogs"),
    #           title="My Diffusion Dog Generator")
    #       app.launch()          ← on Colab: app.launch(share=True)
    pass
