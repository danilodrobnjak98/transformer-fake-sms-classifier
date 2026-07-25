from typing import Dict, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


def load_spam_dataset(config: dict) -> DatasetDict:
  """Load the MENTHOS spam dataset from Hugging Face."""
  dataset = load_dataset(config["dataset_name"])

  if config.get("max_train_samples"):
    dataset["train"] = dataset["train"].select(range(config["max_train_samples"]))
  if config.get("max_eval_samples"):
    dataset["validation"] = dataset["validation"].select(range(config["max_eval_samples"]))
  if config.get("max_test_samples"):
    dataset["test"] = dataset["test"].select(range(config["max_test_samples"]))

  return dataset


def tokenize_dataset(
  dataset: DatasetDict,
  tokenizer: PreTrainedTokenizerBase,
  max_length: int,
) -> DatasetDict:
  """Tokenize text examples for sequence classification."""

  def preprocess(batch: Dict[str, list]) -> Dict[str, list]:
    return tokenizer(
      batch["text"],
      truncation=True,
      padding="max_length",
      max_length=max_length,
    )

  tokenized = dataset.map(preprocess, batched=True)
  tokenized = tokenized.rename_column("label", "labels")
  tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
  return tokenized


def get_label_names(config: dict) -> Tuple[str, str]:
  names = config["label_names"]
  return names[0], names[1]
