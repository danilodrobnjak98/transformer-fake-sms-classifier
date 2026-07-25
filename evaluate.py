import json
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from transformers import EvalPrediction


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)

  return {
    "accuracy": accuracy_score(labels, predictions),
    "precision": precision_score(labels, predictions, average="binary", zero_division=0),
    "recall": recall_score(labels, predictions, average="binary", zero_division=0),
    "f1": f1_score(labels, predictions, average="binary", zero_division=0),
  }


def save_test_report(
  labels: np.ndarray,
  predictions: np.ndarray,
  label_names: tuple[str, str],
  output_path: Path,
) -> Dict[str, float]:
  report = classification_report(
    labels,
    predictions,
    target_names=list(label_names),
    output_dict=True,
    zero_division=0,
  )

  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
  return report
