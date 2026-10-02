"""Grad-CAM + error analysis on the test set.

Run (needs results/<model>.pt and results/<model>_test.json):
  python -m src.gradcam --model efficientnet_b0
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data import (CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, load_arrays,
                      to_rgb01)
from src.train_cnn import build


def target_layer(model, name):
    if name.startswith("resnet"):
        return model.layer4[-1]
    return model.features[-1]  # efficientnet_b0


def gradcam(model, layer, x, cls):
    """x: (1,3,H,W) normalized. Returns CAM in [0,1] (H,W) and class probs."""
    store = {}

    def fwd_hook(_, __, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = layer.register_forward_hook(fwd_hook)
    model.zero_grad()
    with torch.enable_grad():
        logits = model(x)
        logits[0, cls].backward()
    h.remove()
    w = store["g"].mean(dim=(2, 3), keepdim=True)          # channel weights
    cam = torch.relu((w * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                        align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), logits.softmax(1)[0].detach().cpu().numpy()


def prep(img, device):
    rgb = to_rgb01(img)
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
    x = ((x - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)
    return rgb, x


def plot_grid(items, model, layer, X, y, device, title, path):
    n = len(items)
    fig, ax = plt.subplots(2, n, figsize=(2.6 * n, 5.6))
    ax = np.atleast_2d(ax).reshape(2, n)
    for c, i in enumerate(items):
        rgb, x = prep(X[i], device)
        with torch.no_grad():
            pred = int(model(x).argmax(1))
        cam, probs = gradcam(model, layer, x, pred)
        ax[0, c].imshow(rgb)
        ax[1, c].imshow(rgb)
        ax[1, c].imshow(cam, cmap="jet", alpha=0.4)
        ax[0, c].set_title(f"#{i}\ntrue {CLASS_NAMES[y[i]]}\n"
                           f"pred {CLASS_NAMES[pred]} ({probs[pred]:.2f})",
                           fontsize=8)
        ax[0, c].axis("off")
        ax[1, c].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, X, y = load_arrays()
    model, _ = build(a.model)
    model.load_state_dict(torch.load(f"results/{a.model}.pt", map_location=device))
    model.to(device).eval()
    layer = target_layer(model, a.model)

    wrong = json.load(open(f"results/{a.model}_test.json"))["misclassified_test_idx"]
    # rank mistakes by how confident the model was in the wrong class
    conf = []
    for i in wrong:
        _, x = prep(X[i], device)
        with torch.no_grad():
            conf.append(model(x).softmax(1).max().item())
    order = np.argsort(conf)[::-1]
    worst = [wrong[k] for k in order[:a.n]]

    rng = np.random.default_rng(0)
    right = np.setdiff1d(np.arange(len(y)), wrong)
    ok = rng.choice(right, a.n, replace=False).tolist()

    plot_grid(worst, model, layer, X, y, device,
              f"{a.model}: most confident errors",
              f"results/gradcam_{a.model}_errors.png")
    plot_grid(ok, model, layer, X, y, device,
              f"{a.model}: random correct predictions",
              f"results/gradcam_{a.model}_correct.png")
    print(f"{len(wrong)} test errors out of {len(y)}")


if __name__ == "__main__":
    main()