import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config, get_output_dir
from dataset import (
  SpamDataset,
  get_label_names,
  get_or_build_tokenizer,
  load_spam_dataset,
)
from evaluate import compute_metrics_from_arrays, save_test_report
from model import get_classifier


def set_seed(seed: int) -> None:
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: str, criterion: nn.Module):
  model.eval()
  total_loss = 0.0
  all_preds = []
  all_labels = []

  for batch in loader:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids)
    loss = criterion(logits, labels)
    total_loss += loss.item() * labels.size(0)

    preds = torch.argmax(logits, dim=-1)
    all_preds.append(preds.cpu().numpy())
    all_labels.append(labels.cpu().numpy())

  preds_np = np.concatenate(all_preds)
  labels_np = np.concatenate(all_labels)
  metrics = compute_metrics_from_arrays(labels_np, preds_np)
  metrics["loss"] = total_loss / max(len(loader.dataset), 1)
  return metrics, labels_np, preds_np


def train_model(config: dict | None = None) -> None:
  config = config or get_config()
  set_seed(config["seed"])

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using device: {device}")
  print("Training encoder transformer FROM SCRATCH (no DistilBERT / no pretrained weights).")

  raw_dataset = load_spam_dataset(config)
  print(raw_dataset)
  print("Example:", raw_dataset["train"][0])

  tokenizer = get_or_build_tokenizer(config, raw_dataset["train"])
  pad_id = tokenizer.token_to_id("[PAD]")

  train_ds = SpamDataset(raw_dataset["train"], tokenizer, config["context_size"])
  val_ds = SpamDataset(raw_dataset["validation"], tokenizer, config["context_size"])
  test_ds = SpamDataset(raw_dataset["test"], tokenizer, config["context_size"])

  train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=config["eval_batch_size"], shuffle=False)
  test_loader = DataLoader(test_ds, batch_size=config["eval_batch_size"], shuffle=False)

  model = get_classifier(config, tokenizer.get_vocab_size(), pad_id).to(device)
  print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
  )

  output_dir = get_output_dir(config)
  output_dir.mkdir(parents=True, exist_ok=True)
  best_dir = output_dir / "best_model"
  best_dir.mkdir(parents=True, exist_ok=True)

  history = []
  best_f1 = -1.0

  for epoch in range(1, config["num_epochs"] + 1):
    model.train()
    running_loss = 0.0

    progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config['num_epochs']}")
    for batch in progress:
      input_ids = batch["input_ids"].to(device)
      labels = batch["labels"].to(device)

      optimizer.zero_grad()
      logits = model(input_ids)
      loss = criterion(logits, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * labels.size(0)
      progress.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = running_loss / max(len(train_ds), 1)
    val_metrics, _, _ = evaluate_loader(model, val_loader, device, criterion)

    epoch_log = {
      "epoch": epoch,
      "train_loss": train_loss,
      "eval_loss": val_metrics["loss"],
      "eval_accuracy": val_metrics["accuracy"],
      "eval_precision": val_metrics["precision"],
      "eval_recall": val_metrics["recall"],
      "eval_f1": val_metrics["f1"],
    }
    history.append(epoch_log)

    print(
      f"Epoch {epoch}: train_loss={train_loss:.4f} "
      f"val_loss={val_metrics['loss']:.4f} "
      f"val_acc={val_metrics['accuracy']:.4f} "
      f"val_f1={val_metrics['f1']:.4f}"
    )

    if val_metrics["f1"] > best_f1:
      best_f1 = val_metrics["f1"]
      torch.save(
        {
          "model_state_dict": model.state_dict(),
          "config": config,
          "vocab_size": tokenizer.get_vocab_size(),
          "pad_token_id": pad_id,
        },
        best_dir / "model.pt",
      )
      tokenizer.save(str(best_dir / "tokenizer.json"))
      print(f"  -> saved new best model (f1={best_f1:.4f})")

  (output_dir / "metrics_history.json").write_text(
    json.dumps(
      {
        "log_history": history,
        "epoch": config["num_epochs"],
        "num_train_epochs": config["num_epochs"],
        "best_eval_f1": best_f1,
      },
      indent=2,
    ),
    encoding="utf-8",
  )

  # Load best checkpoint for final test evaluation
  checkpoint = torch.load(best_dir / "model.pt", map_location=device, weights_only=False)
  model.load_state_dict(checkpoint["model_state_dict"])

  test_metrics, test_labels, test_preds = evaluate_loader(model, test_loader, device, criterion)
  print("Test metrics:", test_metrics)

  label_names = get_label_names(config)
  report = save_test_report(
    test_labels,
    test_preds,
    label_names,
    output_dir / "test_classification_report.json",
  )
  print(
    {
      "ham": report["ham"],
      "spam": report["spam"],
      "accuracy": report["accuracy"],
      "macro avg": report["macro avg"],
      "weighted avg": report["weighted avg"],
    }
  )


if __name__ == "__main__":
  warnings.filterwarnings("ignore")
  train_model()
