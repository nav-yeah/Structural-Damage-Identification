"""Web UI: upload an image -> prediction + Grad-CAM overlay.

  python -m src.app
  python -m src.app --model resnet50
Then open the printed local URL (http://127.0.0.1:7860).
"""
import argparse

import gradio as gr
import numpy as np
import torch
from matplotlib import cm
from PIL import Image

from src.data import CLASS_NAMES
from src.demo import load_model, to_tensor
from src.gradcam import gradcam, target_layer


def make_predict(model, layer):
    def predict(img):
        if img is None:
            return None, None
        pil = Image.fromarray(img).convert("RGB").resize((224, 224))
        rgb = np.asarray(pil, dtype=np.float32) / 255.0
        x = to_tensor(rgb)
        with torch.no_grad():
            pred = int(model(x).argmax(1))
        cam, probs = gradcam(model, layer, x, pred)
        heat = cm.jet(cam)[..., :3]
        overlay = np.clip(0.6 * rgb + 0.4 * heat, 0, 1)
        overlay = (overlay * 255).astype(np.uint8)
        return {CLASS_NAMES[i]: float(probs[i]) for i in range(2)}, overlay

    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0",
                    choices=["resnet18", "resnet50", "efficientnet_b0"])
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()

    model = load_model(a.model, a.ckpt or f"results/{a.model}.pt")
    layer = target_layer(model, a.model)

    ui = gr.Interface(
        fn=make_predict(model, layer),
        inputs=gr.Image(type="numpy", label="Upload a photo of a structure"),
        outputs=[
            gr.Label(num_top_classes=2, label="Prediction"),
            gr.Image(label="Grad-CAM (where the model looked)"),
        ],
        title="Structural Damage Recognition",
        description=(f"{a.model} fine-tuned on PEER Hub ImageNet Task 2 "
                     "(Damaged vs. Undamaged). Images are resized to 224x224. "
                     "Intended for photos of buildings and structural elements."),
    )
    ui.launch()  # local only; do not use share=True


if __name__ == "__main__":
    main()