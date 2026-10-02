"""Transfer learning for damage-state classification.

Phase 1: freeze backbone, train the new head.
Phase 2: unfreeze everything, fine-tune at a lower LR.
Best epoch is chosen on validation; test is scored once at the end.

Run:  python -m src.train_cnn --model resnet18 --epochs_head 1 --epochs_ft 1   (smoke test)
      python -m src.train_cnn --model resnet50
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models
from tqdm import tqdm

from src.data import CLASS_NAMES, get_loaders


def build(name):
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 2)
        head = m.fc
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 2)
        head = m.fc
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        head = m.classifier
    else:
        raise ValueError(name)
    return m, head


def set_frozen(model, head, frozen):
    for p in model.parameters():
        p.requires_grad = not frozen
    if frozen:
        for p in head.parameters():
            p.requires_grad = True


def run_epoch(model, head, loader, device, opt=None, frozen=False, desc=""):
    train = opt is not None
    model.train(train)
    if train and frozen:  # keep backbone BatchNorm stats fixed
        model.eval()
        head.train()
    loss_fn = nn.CrossEntropyLoss()
    tot_loss, correct, n = 0.0, 0, 0
    preds, labels = [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot_loss += loss.item() * len(y)
            p = out.argmax(1)
            correct += (p == y).sum().item()
            n += len(y)
            preds.append(p.cpu().numpy())
            labels.append(y.cpu().numpy())
    return tot_loss / n, correct / n, np.concatenate(preds), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50",
                    choices=["resnet18", "resnet50", "efficientnet_b0"])
    ap.add_argument("--epochs_head", type=int, default=3)
    ap.add_argument("--epochs_ft", type=int, default=7)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_ft", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    train_dl, val_dl, test_dl = get_loaders(batch_size=a.batch,
                                            num_workers=a.workers)
    model, head = build(a.model)
    model.to(device)

    history, best_val, ckpt = [], -1.0, f"results/{a.model}.pt"
    phases = [("head", a.epochs_head, a.lr_head, True),
              ("finetune", a.epochs_ft, a.lr_ft, False)]
    ep = 0
    for phase, n_ep, lr, frozen in phases:
        if n_ep == 0:
            continue
        set_frozen(model, head, frozen)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr)
        for _ in range(n_ep):
            ep += 1
            t = time.time()
            tl, ta, _, _ = run_epoch(model, head, train_dl, device, opt, frozen,
                                     f"ep{ep} train")
            vl, va, _, _ = run_epoch(model, head, val_dl, device,
                                     desc=f"ep{ep} val")
            history.append(dict(epoch=ep, phase=phase, train_loss=tl,
                                train_acc=ta, val_loss=vl, val_acc=va))
            print(f"[{phase}] ep{ep}: train {ta:.4f}/{tl:.4f}  "
                  f"val {va:.4f}/{vl:.4f}  ({time.time() - t:.0f}s)")
            if va > best_val:
                best_val = va
                torch.save(model.state_dict(), ckpt)
            with open(f"results/{a.model}_history.json", "w") as f:
                json.dump(history, f, indent=2)

    # single test evaluation with the best-validation checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, test_acc, pred, true = run_epoch(model, head, test_dl, device, desc="test")
    print(f"\n{a.model}: best val {best_val:.4f}  TEST acc {test_acc:.4f}")
    print(classification_report(true, pred, target_names=CLASS_NAMES, digits=3))
    print("confusion matrix [rows=true]:\n", confusion_matrix(true, pred))
    with open(f"results/{a.model}_test.json", "w") as f:
        json.dump(dict(
            best_val_acc=best_val, test_acc=test_acc,
            report=classification_report(true, pred, target_names=CLASS_NAMES,
                                         output_dict=True),
            confusion=confusion_matrix(true, pred).tolist(),
            misclassified_test_idx=np.where(pred != true)[0].tolist(),
        ), f, indent=2)


if __name__ == "__main__":
    main()