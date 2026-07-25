import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_classifier(model_dir: str):
  tokenizer = AutoTokenizer.from_pretrained(model_dir)
  model = AutoModelForSequenceClassification.from_pretrained(model_dir)
  model.eval()
  return tokenizer, model


def predict_text(text: str, model_dir: str) -> dict:
  tokenizer, model = load_classifier(model_dir)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)

  encoded = tokenizer(
    text,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt",
  )
  encoded = {key: value.to(device) for key, value in encoded.items()}

  with torch.no_grad():
    logits = model(**encoded).logits
    probabilities = torch.softmax(logits, dim=-1)[0]

  label_id = int(torch.argmax(probabilities).item())
  label_name = "spam" if label_id == 1 else "ham"

  return {
    "label": label_name,
    "label_id": label_id,
    "confidence": float(probabilities[label_id].item()),
    "probabilities": {
      "ham": float(probabilities[0].item()),
      "spam": float(probabilities[1].item()),
    },
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Classify a message as ham or spam.")
  parser.add_argument("--text", required=True, help="Message text to classify.")
  parser.add_argument(
    "--model-dir",
    default="outputs/spam_classifier/best_model",
    help="Path to the fine-tuned model directory.",
  )
  args = parser.parse_args()

  if not Path(args.model_dir).exists():
    raise FileNotFoundError(
      f"Model not found at '{args.model_dir}'. Train first with: python train.py"
    )

  result = predict_text(args.text, args.model_dir)
  print(result)


if __name__ == "__main__":
  main()
