import json
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def compute_metrics_from_arrays(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
  return {
    "accuracy": float(accuracy_score(labels, predictions)),
    "precision": float(precision_score(labels, predictions, average="binary", zero_division=0)),
    "recall": float(recall_score(labels, predictions, average="binary", zero_division=0)),
    "f1": float(f1_score(labels, predictions, average="binary", zero_division=0)),
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
