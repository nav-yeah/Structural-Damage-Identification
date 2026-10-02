"""Live demo: classify test images or your own photos, with Grad-CAM overlay.

  python -m src.demo --idx 391 449 1232 44
  python -m src.demo --image path\\to\\photo.jpg other.jpg
  python -m src.demo --idx 391 --model resnet50 --open
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from src.data import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, to_rgb01
from src.gradcam import gradcam, target_layer


def load_model(name, ckpt):
    """Build the architecture without pretrained weights, then load ours."""
    if name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    else:
        raise ValueError(name)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.eval()


def to_tensor(rgb01):
    x = torch.from_numpy(np.ascontiguousarray(rgb01)).permute(2, 0, 1)
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)


def load_photo(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    return np.asarray(img, dtype=np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0",
                    choices=["resnet18", "resnet50", "efficientnet_b0"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--idx", type=int, nargs="*", default=[],
                    help="indices into the test set")
    ap.add_argument("--image", nargs="*", default=[], help="paths to photos")
    ap.add_argument("--data_dir", default="data/task2")
    ap.add_argument("--out", default="results/demo.png")
    ap.add_argument("--open", action="store_true",
                    help="open the saved figure (Windows)")
    a = ap.parse_args()
    if not a.idx and not a.image:
        ap.error("give --idx and/or --image")

    ckpt = a.ckpt or f"results/{a.model}.pt"
    model = load_model(a.model, ckpt)
    layer = target_layer(model, a.model)

    items = []  # (label, rgb01 image, true class or None)
    if a.idx:
        X = np.load(f"{a.data_dir}/task2_X_test.npy", mmap_mode="r")
        y = np.load(f"{a.data_dir}/task2_y_test.npy").argmax(1)
        for i in a.idx:
            items.append((f"test #{i}", to_rgb01(X[i]), int(y[i])))
    for p in a.image:
        items.append((os.path.basename(p), load_photo(p), None))

    fig, ax = plt.subplots(2, len(items), figsize=(2.8 * len(items), 6))
    ax = np.asarray(ax).reshape(2, len(items))
    print(f"model: {a.model}  checkpoint: {ckpt}\n")
    for c, (name, rgb, true) in enumerate(items):
        x = to_tensor(rgb)
        t = time.time()
        with torch.no_grad():
            pred = int(model(x).argmax(1))
        cam, probs = gradcam(model, layer, x, pred)
        dt = time.time() - t
        verdict = "" if true is None else (
            "  [correct]" if pred == true else "  [WRONG]")
        truth = "" if true is None else f"true={CLASS_NAMES[true]}  "
        print(f"{name}: {truth}pred={CLASS_NAMES[pred]} "
              f"(Damaged {probs[0]:.2f} / Undamaged {probs[1]:.2f})"
              f"{verdict}  [{dt:.2f}s]")
        ax[0, c].imshow(rgb)
        ax[1, c].imshow(rgb)
        ax[1, c].imshow(cam, cmap="jet", alpha=0.4)
        title = f"{name}\n" + (f"true {CLASS_NAMES[true]}\n" if true is not None else "")
        ax[0, c].set_title(title + f"pred {CLASS_NAMES[pred]} ({probs[pred]:.2f})",
                           fontsize=8)
        ax[0, c].axis("off")
        ax[1, c].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    plt.savefig(a.out, dpi=130)
    print(f"\nsaved {a.out}")
    if a.open and hasattr(os, "startfile"):
        os.startfile(os.path.abspath(a.out))


if __name__ == "__main__":
    main()