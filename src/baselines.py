"""Classical baselines: kNN, logistic regression, RBF-SVM on 28x28 thumbnails.

Hyperparameters are picked on the validation split; test is evaluated once.
Run:  python -m src.baselines
"""
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import CAFFE_MEAN_BGR, CLASS_NAMES, load_arrays, split_indices

CACHE = "results/baseline_features.npz"
BLOCK = 8  # 224 / 8 = 28


def thumbnails(X, chunk=256):
    """(N,224,224,3) BGR mean-subtracted -> (N, 28*28*3) RGB in [0,1]."""
    out = []
    for s in range(0, len(X), chunk):
        x = np.asarray(X[s:s + chunk], dtype=np.float32) + CAFFE_MEAN_BGR
        x = np.clip(x[..., ::-1] / 255.0, 0, 1)
        n, h, w, c = x.shape
        x = x.reshape(n, h // BLOCK, BLOCK, w // BLOCK, BLOCK, c).mean((2, 4))
        out.append(x.reshape(n, -1))
    return np.concatenate(out)


def get_features():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    os.makedirs("results", exist_ok=True)
    X_tr, y_tr, X_te, y_te = load_arrays()
    print("Extracting thumbnails (one-time, a few minutes)...")
    Xtr, Xte = thumbnails(X_tr), thumbnails(X_te)
    np.savez_compressed(CACHE, Xtr=Xtr, ytr=y_tr, Xte=Xte, yte=y_te)
    return Xtr, y_tr, Xte, y_te


def tune_and_eval(name, make_model, grid, Xtr, ytr, Xva, yva, Xte, yte):
    best = (-1, None)
    for p in grid:
        t = time.time()
        m = make_model(p).fit(Xtr, ytr)
        acc = accuracy_score(yva, m.predict(Xva))
        print(f"  {name} {p}: val acc {acc:.4f} ({time.time() - t:.0f}s)")
        if acc > best[0]:
            best = (acc, p)
    val_acc, p = best
    # refit on train+val with the chosen hyperparameter, then score test once
    m = make_model(p).fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    pred = m.predict(Xte)
    print(f"\n{name} best={p} val={val_acc:.4f} TEST acc={accuracy_score(yte, pred):.4f}")
    print(classification_report(yte, pred, target_names=CLASS_NAMES, digits=3))
    print("confusion matrix [rows=true]:\n", confusion_matrix(yte, pred), "\n")
    return {
        "best_param": p,
        "val_acc": val_acc,
        "test_acc": accuracy_score(yte, pred),
        "report": classification_report(yte, pred, target_names=CLASS_NAMES,
                                        output_dict=True),
    }


def main():
    Xall, yall, Xte, yte = get_features()
    tr, va = split_indices(yall)
    Xtr, ytr, Xva, yva = Xall[tr], yall[tr], Xall[va], yall[va]
    print("features:", Xtr.shape, Xva.shape, Xte.shape)

    results = {}
    results["kNN"] = tune_and_eval(
        "kNN",
        lambda k: make_pipeline(StandardScaler(),
                                KNeighborsClassifier(n_neighbors=k, n_jobs=-1)),
        [1, 5, 11, 21], Xtr, ytr, Xva, yva, Xte, yte)
    results["LogReg"] = tune_and_eval(
        "LogReg",
        lambda C: make_pipeline(StandardScaler(),
                                LogisticRegression(C=C, max_iter=500)),
        [0.0001, 0.001, 0.01, 0.1], Xtr, ytr, Xva, yva, Xte, yte)
    results["SVM-RBF"] = tune_and_eval(
        "SVM-RBF",
        lambda C: make_pipeline(StandardScaler(), SVC(C=C, kernel="rbf")),
        [0.1, 1, 10], Xtr, ytr, Xva, yva, Xte, yte)

    with open("results/baselines.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("Saved results/baselines.json")


if __name__ == "__main__":
    main()