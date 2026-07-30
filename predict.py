import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import get_config
from dataset import encode_text
from model import get_classifier


def load_classifier(model_dir: str):
  model_dir = Path(model_dir)
  checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
  tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
  config = checkpoint.get("config") or get_config()

  model = get_classifier(
    config,
    checkpoint["vocab_size"],
    checkpoint["pad_token_id"],
  )
  model.load_state_dict(checkpoint["model_state_dict"])
  model.eval()
  return tokenizer, model, config


def predict_text(text: str, model_dir: str) -> dict:
  tokenizer, model, config = load_classifier(model_dir)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)

  input_ids = encode_text(tokenizer, text, config["context_size"]).unsqueeze(0).to(device)
  with torch.no_grad():
    logits = model(input_ids)
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
    default="outputs/spam_classifier_scratch/best_model",
    help="Path to the trained-from-scratch model directory.",
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
