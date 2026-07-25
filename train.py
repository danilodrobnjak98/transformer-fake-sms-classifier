import random
import warnings

import numpy as np
import torch
from transformers import (
  AutoModelForSequenceClassification,
  AutoTokenizer,
  Trainer,
  TrainingArguments,
  set_seed,
)

from config import get_config, get_output_dir
from dataset import get_label_names, load_spam_dataset, tokenize_dataset
from evaluate import compute_metrics, save_test_report


def train_model(config: dict | None = None) -> None:
  config = config or get_config()
  set_seed(config["seed"])

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using device: {device}")

  raw_dataset = load_spam_dataset(config)
  print(raw_dataset)
  print("Example:", raw_dataset["train"][0])

  tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
  tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer, config["max_length"])

  model = AutoModelForSequenceClassification.from_pretrained(
    config["model_name"],
    num_labels=config["num_labels"],
  )

  output_dir = get_output_dir(config)
  output_dir.mkdir(parents=True, exist_ok=True)

  training_args = TrainingArguments(
    output_dir=str(output_dir),
    num_train_epochs=config["num_epochs"],
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["eval_batch_size"],
    learning_rate=config["learning_rate"],
    weight_decay=config["weight_decay"],
    warmup_ratio=config["warmup_ratio"],
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
    seed=config["seed"],
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
  )

  trainer.train()
  trainer.save_model(str(output_dir / "best_model"))
  tokenizer.save_pretrained(str(output_dir / "best_model"))

  validation_metrics = trainer.evaluate()
  print("Validation metrics:", validation_metrics)

  test_predictions = trainer.predict(tokenized_dataset["test"])
  test_labels = test_predictions.label_ids
  test_preds = np.argmax(test_predictions.predictions, axis=-1)

  label_names = get_label_names(config)
  report = save_test_report(
    test_labels,
    test_preds,
    label_names,
    output_dir / "test_classification_report.json",
  )

  print("Test classification report:")
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
