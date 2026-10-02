"""Learning curves and results table from saved JSON results.

  python -m src.plots
Outputs: results/learning_curves.png, results/results_table.md, results/results_table.csv
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [("resnet18", "ResNet18"), ("resnet50", "ResNet50"),
          ("efficientnet_b0", "EfficientNet-B0")]
BASELINES = [("kNN", "kNN"), ("LogReg", "Logistic regression"),
             ("SVM-RBF", "SVM (RBF)")]


def load(path):
    with open(path) as f:
        return json.load(f)


def learning_curves():
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5), sharex="col")
    for c, (key, label) in enumerate(MODELS):
        h = load(f"results/{key}_history.json")
        ep = [r["epoch"] for r in h]
        boundary = next((r["epoch"] for r in h if r["phase"] == "finetune"), None)
        best = max(h, key=lambda r: r["val_acc"])
        for row, (metric, ylabel) in enumerate([("acc", "Accuracy"),
                                                ("loss", "Loss")]):
            a = ax[row, c]
            a.plot(ep, [r[f"train_{metric}"] for r in h], "o-", label="train")
            a.plot(ep, [r[f"val_{metric}"] for r in h], "s-", label="val")
            if boundary:
                a.axvline(boundary - 0.5, color="gray", ls="--", lw=1)
            if metric == "acc":
                a.plot(best["epoch"], best["val_acc"], "k*", ms=12,
                       label=f"best val ({best['val_acc']:.3f})")
            a.set_ylabel(ylabel)
            a.grid(alpha=0.3)
            if row == 0:
                a.set_title(label)
                a.legend(fontsize=8)
            if row == 1:
                a.set_xlabel("epoch (dashed line: head -> fine-tune)")
    plt.tight_layout()
    plt.savefig("results/learning_curves.png", dpi=150)
    plt.close(fig)
    print("saved results/learning_curves.png")


def results_table():
    rows = []
    base = load("results/baselines.json")
    for key, label in BASELINES:
        r = base[key]
        rows.append((label, r["best_param"], r["val_acc"], r["test_acc"],
                     r["report"]))
    for key, label in MODELS:
        r = load(f"results/{key}_test.json")
        rows.append((label, "-", r["best_val_acc"], r["test_acc"], r["report"]))

    header = ["Model", "Tuned param", "Val acc", "Test acc", "Macro F1",
              "Recall (Damaged)", "Recall (Undamaged)"]
    lines = []
    for label, p, va, ta, rep in rows:
        lines.append([label, str(p), f"{va:.3f}", f"{ta:.3f}",
                      f"{rep['macro avg']['f1-score']:.3f}",
                      f"{rep['Damaged']['recall']:.3f}",
                      f"{rep['Undamaged']['recall']:.3f}"])

    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    md += ["| " + " | ".join(l) + " |" for l in lines]
    with open("results/results_table.md", "w") as f:
        f.write("\n".join(md) + "\n")
    with open("results/results_table.csv", "w") as f:
        f.write(",".join(header) + "\n")
        f.writelines(",".join(f'"{x}"' for x in l) + "\n" for l in lines)
    print("\n".join(md))
    print("\nsaved results/results_table.md and .csv")


if __name__ == "__main__":
    learning_curves()
    results_table()