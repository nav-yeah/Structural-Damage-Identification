"""Data loading for PEER Hub ImageNet Task 2 (damage state).

Labels: 0 = Damaged, 1 = Undamaged.
Stored arrays are float32, BGR, ImageNet-channel-mean subtracted
(Keras 'caffe' preprocessing). to_rgb01() converts to RGB in [0, 1].
Arrays are memory-mapped (X_train is ~7 GB).
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["Damaged", "Undamaged"]
CAFFE_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_rgb01(img):
    """Stored (H,W,3) BGR mean-subtracted -> (H,W,3) RGB float32 in [0,1]."""
    x = np.asarray(img, dtype=np.float32) + CAFFE_MEAN_BGR
    x = x[..., ::-1]  # BGR -> RGB
    return np.clip(x / 255.0, 0.0, 1.0).astype(np.float32)


def load_arrays(data_dir="data/task2"):
    X_tr = np.load(f"{data_dir}/task2_X_train.npy", mmap_mode="r")
    y_tr = np.load(f"{data_dir}/task2_y_train.npy").argmax(1)
    X_te = np.load(f"{data_dir}/task2_X_test.npy", mmap_mode="r")
    y_te = np.load(f"{data_dir}/task2_y_test.npy").argmax(1)
    assert len(X_tr) == len(y_tr), f"train mismatch: {len(X_tr)} vs {len(y_tr)}"
    assert len(X_te) == len(y_te), f"test mismatch: {len(X_te)} vs {len(y_te)}"
    return X_tr, y_tr, X_te, y_te


def split_indices(y_train, val_frac=0.1, seed=42):
    """Stratified train/val split of the official training set."""
    idx = np.arange(len(y_train))
    tr, va = train_test_split(
        idx, test_size=val_frac, stratify=y_train, random_state=seed
    )
    return np.sort(tr), np.sort(va)


class DamageDataset(Dataset):
    def __init__(self, X, y, indices=None, augment=False):
        self.X, self.y = X, y
        self.idx = np.arange(len(y)) if indices is None else indices
        self.augment = augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = torch.from_numpy(np.ascontiguousarray(to_rgb01(self.X[j])))
        x = x.permute(2, 0, 1)  # HWC -> CHW
        if self.augment and torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[2])  # horizontal flip only (damage-safe)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, int(self.y[j])


def get_loaders(data_dir="data/task2", batch_size=32, val_frac=0.1,
                seed=42, num_workers=0):
    X_tr, y_tr, X_te, y_te = load_arrays(data_dir)
    tr_idx, va_idx = split_indices(y_tr, val_frac, seed)
    train = DamageDataset(X_tr, y_tr, tr_idx, augment=True)
    val = DamageDataset(X_tr, y_tr, va_idx)
    test = DamageDataset(X_te, y_te)
    kw = dict(batch_size=batch_size, num_workers=num_workers)
    return (
        DataLoader(train, shuffle=True, **kw),
        DataLoader(val, **kw),
        DataLoader(test, **kw),
    )


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = load_arrays()
    tr, va = split_indices(y_tr)
    print("train", X_tr.shape, np.bincount(y_tr))
    print("test ", X_te.shape, np.bincount(y_te))
    print("split", len(tr), "train /", len(va), "val")

    sample = np.asarray(X_tr[:200], dtype=np.float32)
    print("raw per-channel min:", sample.min(axis=(0, 1, 2)))
    print("raw per-channel max:", sample.max(axis=(0, 1, 2)))
    rec = sample + CAFFE_MEAN_BGR
    print("recovered min/max  :", rec.min(), rec.max(), "(expect ~0 and ~255)")