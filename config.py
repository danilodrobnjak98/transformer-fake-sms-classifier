from pathlib import Path

# https://huggingface.co/datasets/LHRS-UM-FERI/MENTHOS-dataset-spam

def get_config() -> dict:
  """Configuration for SMS spam classification with an encoder trained from scratch."""
  return {
    "dataset_name": "LHRS-UM-FERI/MENTHOS-dataset-spam",
    "batch_size": 64,
    "eval_batch_size": 128,
    "num_epochs": 15,
    "learning_rate": 3e-4, #AdamW param
    "weight_decay": 0.01, #AdamW param
    "context_size": 128,
    "model_dimension": 128, #encoder
    "number_of_blocks": 2, #encoder
    "heads": 4, #encoder
    "feed_forward_dimension": 512, #encoder
    "dropout": 0.1, #encoder
    "tokenizer_file": "outputs/spam_classifier_scratch/tokenizer.json",
    "tokenizer_min_frequency": 2,
    "output_dir": "outputs/spam_classifier_scratch",
    "seed": 42, #default number
    "num_labels": 2,
    "label_names": ["ham", "spam"],
    "max_train_samples": None,
    "max_eval_samples": None,
    "max_test_samples": None,
  }


def get_output_dir(config: dict) -> Path:
  return Path(config["output_dir"])
