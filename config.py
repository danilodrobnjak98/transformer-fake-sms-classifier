from pathlib import Path


def get_config() -> dict:
  """Configuration for SMS/email spam classification with DistilBERT."""
  return {
    "dataset_name": "LHRS-UM-FERI/MENTHOS-dataset-spam",
    "model_name": "distilbert-base-uncased",
    "max_length": 128,
    "batch_size": 32,
    "eval_batch_size": 64,
    "num_epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "output_dir": "outputs/spam_classifier",
    "seed": 42,
    "num_labels": 2,
    "label_names": ["ham", "spam"],
    # For a quick smoke test on CPU, set e.g. 1000 / 500 / 500.
    # For the full MENTHOS run, keep all three values as None.
    "max_train_samples": None,
    "max_eval_samples": None,
    "max_test_samples": None,
  }


def get_output_dir(config: dict) -> Path:
  return Path(config["output_dir"])
