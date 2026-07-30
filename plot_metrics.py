"""Generate training/validation and test plots from from-scratch training logs."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HISTORY_PATH = Path("outputs/spam_classifier_scratch/metrics_history.json")
TEST_REPORT_PATH = Path("outputs/spam_classifier_scratch/test_classification_report.json")
OUT_DIR = Path("outputs/spam_classifier_scratch/plots")
STEPS_PER_EPOCH = 953  # ceil(60982 / 64)


def plot_test_metrics() -> None:
  report = json.loads(TEST_REPORT_PATH.read_text(encoding="utf-8"))

  metrics = ["precision", "recall", "f1-score"]
  metric_labels = ["Precision", "Recall", "F1-score"]
  ham = [report["ham"][m] for m in metrics]
  spam = [report["spam"][m] for m in metrics]
  x = np.arange(len(metrics))
  width = 0.35

  # Slika 5. Precision, Recall, F1 za ham i spam
  fig, ax = plt.subplots(figsize=(8, 4.8))
  b1 = ax.bar(x - width / 2, ham, width, label="Ham", color="#38bdf8")
  b2 = ax.bar(x + width / 2, spam, width, label="Spam", color="#f97316")
  ax.set_ylabel("Vrednost")
  ax.set_title("Slika 5. Precision, Recall, F1 za ham i spam")
  ax.set_xticks(x)
  ax.set_xticklabels(metric_labels)
  ax.set_ylim(0.98, 1.0)
  ax.legend()
  ax.grid(True, axis="y", alpha=0.3)
  for bars in (b1, b2):
    for bar in bars:
      h = bar.get_height()
      ax.annotate(
        f"{h * 100:.2f}%",
        xy=(bar.get_x() + bar.get_width() / 2, h),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
      )
  fig.savefig(OUT_DIR / "05_test_metrics_by_class.png")
  plt.close(fig)

  # Slika 6. Accuracy + macro metrike
  overall_names = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)"]
  overall_vals = [
    report["accuracy"],
    report["macro avg"]["precision"],
    report["macro avg"]["recall"],
    report["macro avg"]["f1-score"],
  ]
  colors = ["#22c55e", "#2563eb", "#a855f7", "#dc2626"]

  fig, ax = plt.subplots(figsize=(8, 4.8))
  bars = ax.bar(overall_names, overall_vals, color=colors, width=0.6)
  ax.set_ylabel("Vrednost")
  ax.set_title("Slika 6. Accuracy + macro metrike")
  ax.set_ylim(0.98, 1.0)
  ax.grid(True, axis="y", alpha=0.3)
  for bar, val in zip(bars, overall_vals):
    ax.annotate(
      f"{val * 100:.2f}%",
      xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
      xytext=(0, 3),
      textcoords="offset points",
      ha="center",
      va="bottom",
      fontsize=10,
      fontweight="bold",
    )
  fig.savefig(OUT_DIR / "06_test_metrics_overall.png")
  plt.close(fig)


def main() -> None:
  history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
  logs = history["log_history"]

  epochs = [e["epoch"] for e in logs]
  train_loss = [e["train_loss"] for e in logs]
  eval_loss = [e["eval_loss"] for e in logs]
  # Approximate global step at end of each epoch (per-step loss was not logged)
  steps = [e["epoch"] * STEPS_PER_EPOCH for e in logs]

  OUT_DIR.mkdir(parents=True, exist_ok=True)
  plt.rcParams.update(
    {
      "font.size": 11,
      "axes.titlesize": 13,
      "axes.labelsize": 11,
      "figure.dpi": 150,
      "savefig.dpi": 200,
      "savefig.bbox": "tight",
    }
  )

  # Slika 1. Train loss po koraku (aproksimacija: loss na kraju epohe)
  fig, ax = plt.subplots(figsize=(8, 4.5))
  ax.plot(steps, train_loss, color="#2563eb", linewidth=2, marker="o", markersize=5, label="Train loss")
  ax.set_xlabel("Korak (step)")
  ax.set_ylabel("Loss")
  ax.set_title("Slika 1. Train loss po koraku")
  ax.grid(True, alpha=0.3)
  ax.legend()
  fig.savefig(OUT_DIR / "01_train_loss_po_koraku.png")
  plt.close(fig)

  # Slika 2. Validation loss po epohi
  fig, ax = plt.subplots(figsize=(8, 4.5))
  ax.plot(
    epochs,
    eval_loss,
    marker="o",
    color="#dc2626",
    linewidth=2,
    markersize=6,
    label="Validation loss",
  )
  ax.set_xticks(epochs)
  ax.set_xlabel("Epoha")
  ax.set_ylabel("Loss")
  ax.set_title("Slika 2. Validation loss po epohi")
  ax.grid(True, alpha=0.3)
  ax.legend()
  fig.savefig(OUT_DIR / "02_val_loss_po_epohi.png")
  plt.close(fig)

  # Slika 3. Train vs validation loss
  fig, ax = plt.subplots(figsize=(8, 4.5))
  ax.plot(epochs, train_loss, color="#2563eb", linewidth=2, marker="o", markersize=5, label="Train loss")
  ax.plot(epochs, eval_loss, color="#dc2626", linewidth=2, marker="s", markersize=5, label="Validation loss")
  ax.set_xticks(epochs)
  ax.set_xlabel("Epoha")
  ax.set_ylabel("Loss")
  ax.set_title("Slika 3. Train vs validation loss")
  ax.grid(True, alpha=0.3)
  ax.legend()
  fig.savefig(OUT_DIR / "03_train_vs_val_loss.png")
  plt.close(fig)

  # Slika 4. Accuracy, Precision, Recall, F1
  fig, ax = plt.subplots(figsize=(8, 4.5))
  ax.plot(epochs, [e["eval_accuracy"] for e in logs], marker="o", linewidth=2, label="Accuracy")
  ax.plot(epochs, [e["eval_precision"] for e in logs], marker="s", linewidth=2, label="Precision")
  ax.plot(epochs, [e["eval_recall"] for e in logs], marker="^", linewidth=2, label="Recall")
  ax.plot(epochs, [e["eval_f1"] for e in logs], marker="D", linewidth=2, label="F1")
  ax.set_xticks(epochs)
  ax.set_xlabel("Epoha")
  ax.set_ylabel("Vrednost metrike")
  ax.set_title("Slika 4. Accuracy, Precision, Recall, F1")
  ax.set_ylim(0.97, 1.0)
  ax.grid(True, alpha=0.3)
  ax.legend()
  fig.savefig(OUT_DIR / "04_val_metrics.png")
  plt.close(fig)

  # Overview 2x2 (za brzi pregled)
  fig, axes = plt.subplots(2, 2, figsize=(11, 8))
  axes[0, 0].plot(steps, train_loss, color="#2563eb", linewidth=1.8, marker="o", markersize=4)
  axes[0, 0].set_title("Train loss po koraku")
  axes[0, 0].set_xlabel("Korak")
  axes[0, 0].set_ylabel("Loss")
  axes[0, 0].grid(True, alpha=0.3)

  axes[0, 1].plot(epochs, eval_loss, marker="o", color="#dc2626", linewidth=2, markersize=5)
  axes[0, 1].set_title("Validation loss po epohi")
  axes[0, 1].set_xlabel("Epoha")
  axes[0, 1].set_ylabel("Loss")
  axes[0, 1].set_xticks(epochs)
  axes[0, 1].grid(True, alpha=0.3)

  axes[1, 0].plot(epochs, train_loss, color="#2563eb", linewidth=1.8, marker="o", markersize=4, label="Train")
  axes[1, 0].plot(epochs, eval_loss, color="#dc2626", linewidth=1.8, marker="s", markersize=4, label="Validation")
  axes[1, 0].set_title("Train vs validation loss")
  axes[1, 0].set_xlabel("Epoha")
  axes[1, 0].set_ylabel("Loss")
  axes[1, 0].set_xticks(epochs)
  axes[1, 0].legend()
  axes[1, 0].grid(True, alpha=0.3)

  axes[1, 1].plot(epochs, [e["eval_accuracy"] for e in logs], marker="o", label="Accuracy")
  axes[1, 1].plot(epochs, [e["eval_precision"] for e in logs], marker="s", label="Precision")
  axes[1, 1].plot(epochs, [e["eval_recall"] for e in logs], marker="^", label="Recall")
  axes[1, 1].plot(epochs, [e["eval_f1"] for e in logs], marker="D", label="F1")
  axes[1, 1].set_title("Validation metrike")
  axes[1, 1].set_xlabel("Epoha")
  axes[1, 1].set_ylabel("Vrednost")
  axes[1, 1].set_xticks(epochs)
  axes[1, 1].set_ylim(0.97, 1.0)
  axes[1, 1].legend(fontsize=8)
  axes[1, 1].grid(True, alpha=0.3)

  fig.suptitle("Encoder transformer (od nule) — trening i validacija", fontsize=14, y=0.995)
  fig.tight_layout()
  fig.savefig(OUT_DIR / "00_training_overview.png")
  plt.close(fig)

  plot_test_metrics()

  print(f"Sacuvano u: {OUT_DIR.resolve()}")
  for path in sorted(OUT_DIR.glob("*.png")):
    print(f"  - {path.name}")


if __name__ == "__main__":
  main()
