# SMS Spam Classifier

Fine-tuned **DistilBERT** model for binary SMS/email spam classification (`ham` / `spam`), trained on the [MENTHOS](https://huggingface.co/datasets/LHRS-UM-FERI/MENTHOS-dataset-spam) dataset.

## Results

On the held-out test set (~13k messages):

| Metric | Value |
|--------|-------|
| Accuracy | **99.28%** |
| Macro F1 | **99.28%** |
| Ham F1 | 99.28% |
| Spam F1 | 99.28% |

## Project structure

| File | Description |
|------|-------------|
| `config.py` | Hyperparameters and paths |
| `dataset.py` | Load & tokenize MENTHOS dataset |
| `train.py` | Training loop (Hugging Face `Trainer`) |
| `evaluate.py` | Metrics and classification report |
| `predict.py` | Classify a single message |
| `generate_results_html.py` | Build `results.html` from training outputs |
| `requirements-spam.txt` | Minimal dependencies for this project |
| `run.bat` | Interactive Windows menu |

## Requirements

- Python 3.10+
- Internet access (first run downloads the model and dataset from Hugging Face)

## Quick start (Windows)

```bat
run.bat
```

Menu options:

1. **Setup** — create `.venv` and install dependencies  
2. **Train** — fine-tune DistilBERT  
3. **Predict** — classify one message  
4. **Results** — open HTML report  

Or step by step:

```bat
setup.bat
train.bat
predict.bat "Congratulations! You won a free iPhone. Click here now!"
results.bat
```

## Quick start (Python)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -r requirements-spam.txt
pip install torch   # or CUDA build from pytorch.org

python train.py
python predict.py --text "Hey, are we still meeting for lunch tomorrow?"
python generate_results_html.py
```

## Prediction

```bash
python predict.py --text "Your message here"
```

Example output:

```python
{
  "label": "spam",
  "label_id": 1,
  "confidence": 0.99,
  "probabilities": {"ham": 0.01, "spam": 0.99}
}
```

Model path (default): `outputs/spam_classifier/best_model`

## Configuration

Edit `config.py`:

| Key | Default | Notes |
|-----|---------|-------|
| `model_name` | `distilbert-base-uncased` | Base model |
| `dataset_name` | `LHRS-UM-FERI/MENTHOS-dataset-spam` | HF dataset |
| `max_length` | `128` | Token limit |
| `batch_size` | `32` | Train batch size |
| `num_epochs` | `3` | Training epochs |
| `learning_rate` | `2e-5` | AdamW LR |
| `max_*_samples` | `None` | Set e.g. `1000` for a quick CPU smoke test |

## Labels

| ID | Name |
|----|------|
| 0 | ham (legitimate) |
| 1 | spam |

## License / data

This project uses third-party models and datasets from Hugging Face. Check their respective licenses before redistribution.
