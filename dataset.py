from pathlib import Path
from typing import Iterator, Tuple

import torch
from datasets import DatasetDict, load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]


def load_spam_dataset(config: dict) -> DatasetDict:
  """Load the MENTHOS spam dataset from Hugging Face (data only, not a model)."""
  dataset = load_dataset(config["dataset_name"])

  if config.get("max_train_samples"):
    dataset["train"] = dataset["train"].select(range(config["max_train_samples"]))
  if config.get("max_eval_samples"):
    dataset["validation"] = dataset["validation"].select(range(config["max_eval_samples"]))
  if config.get("max_test_samples"):
    dataset["test"] = dataset["test"].select(range(config["max_test_samples"]))

  return dataset


def _iter_texts(dataset) -> Iterator[str]:
  for item in dataset:
    yield item["text"]


def get_or_build_tokenizer(config: dict, train_split, force_rewrite: bool = False) -> Tokenizer:
  """Train a WordLevel tokenizer from scratch on the training texts."""
  tokenizer_path = Path(config["tokenizer_file"])
  tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

  if tokenizer_path.exists() and not force_rewrite:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  else:
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
      special_tokens=SPECIAL_TOKENS,
      min_frequency=config.get("tokenizer_min_frequency", 2),
    )
    tokenizer.train_from_iterator(_iter_texts(train_split), trainer=trainer)
    tokenizer.save(str(tokenizer_path))

  print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
  return tokenizer


def encode_text(tokenizer: Tokenizer, text: str, context_size: int) -> torch.Tensor:
  sos = tokenizer.token_to_id("[SOS]")
  eos = tokenizer.token_to_id("[EOS]")
  pad = tokenizer.token_to_id("[PAD]")

  token_ids = tokenizer.encode(text).ids
  # Reserve room for [SOS] and [EOS]
  max_body = context_size - 2
  token_ids = token_ids[:max_body]

  ids = [sos] + token_ids + [eos]
  ids = ids + [pad] * (context_size - len(ids))
  return torch.tensor(ids, dtype=torch.long)


class SpamDataset(Dataset):
  def __init__(self, hf_split, tokenizer: Tokenizer, context_size: int) -> None:
    self.data = hf_split
    self.tokenizer = tokenizer
    self.context_size = context_size

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, index: int) -> dict:
    row = self.data[index]
    return {
      "input_ids": encode_text(self.tokenizer, row["text"], self.context_size),
      "labels": torch.tensor(row["label"], dtype=torch.long),
    }


def get_label_names(config: dict) -> Tuple[str, str]:
  names = config["label_names"]
  return names[0], names[1]
