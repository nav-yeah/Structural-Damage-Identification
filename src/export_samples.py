"""Export test images as JPGs for trying the UI / demo.

  python -m src.export_samples --n 10
Writes to samples/ as test<idx>_<TrueClass>.jpg
"""
import argparse
import os

import numpy as np
from PIL import Image

from src.data import CLASS_NAMES, to_rgb01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="images per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", default="data/task2")
    ap.add_argument("--out", default="samples")
    a = ap.parse_args()

    X = np.load(f"{a.data_dir}/task2_X_test.npy", mmap_mode="r")
    y = np.load(f"{a.data_dir}/task2_y_test.npy").argmax(1)
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for c, name in enumerate(CLASS_NAMES):
        for i in rng.choice(np.where(y == c)[0], a.n, replace=False):
            img = (to_rgb01(X[i]) * 255).astype(np.uint8)
            Image.fromarray(img).save(f"{a.out}/test{i}_{name}.jpg", quality=95)
    print(f"saved {2 * a.n} images to {a.out}/")


if __name__ == "__main__":
    main()