# Transformer Fake SMS Classifier

Binary **ham / spam** text classifier built with a **custom encoder-only Transformer trained from scratch**

The model learns token embeddings and attention layers on the [MENTHOS spam dataset](https://huggingface.co/datasets/LHRS-UM-FERI/MENTHOS-dataset-spam) from Hugging Face, then classifies messages using mean-pooled encoder outputs.

## Features

- Encoder-only Transformer implemented in PyTorch (`model.py`)
- Word-level tokenizer trained from the training split
- Training with AdamW, validation F1 checkpointing, and final test evaluation
- CLI prediction for single messages
- HTML results report and optional metric plots
- Windows batch scripts for setup, train, predict, and results

## Requirements

- Python 3.10+
- Internet access (to download the Hugging Face dataset on first run)
- Optional: CUDA GPU for faster training (`torch` with CUDA in `requirements.txt`)

Minimal dependencies are listed in `requirements-spam.txt`. The full Jupyter/CUDA stack is in `requirements.txt`.

## Quick start (Windows)

From the project root:

```bat
run.bat
```

Or use the scripts menu:

```bat
scripts\run.bat
```

Menu options:

1. **Setup** — create `.venv` and install dependencies  
2. **Train** — run `train.py`  
3. **Predict** — classify a message  
4. **Results** — generate and open `results.html`  

### Manual setup

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-spam.txt
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CUDA, install a CUDA build of PyTorch instead (or use `requirements.txt`).

## Training

```bat
.venv\Scripts\python.exe train.py
```

Or:

```bat
scripts\train.bat
```

What training does:

1. Loads `LHRS-UM-FERI/MENTHOS-dataset-spam` (train / validation / test)
2. Builds or loads a WordLevel tokenizer
3. Trains an encoder classifier from scratch
4. Saves the best checkpoint by validation F1
5. Evaluates on the test set and writes a classification report

### Outputs

| Path | Description |
|------|-------------|
| `outputs/spam_classifier_scratch/best_model/model.pt` | Best model weights + config |
| `outputs/spam_classifier_scratch/best_model/tokenizer.json` | Tokenizer used by the best model |
| `outputs/spam_classifier_scratch/tokenizer.json` | Tokenizer built during training |
| `outputs/spam_classifier_scratch/metrics_history.json` | Per-epoch train/val metrics |
| `outputs/spam_classifier_scratch/test_classification_report.json` | Test classification report |

## Prediction

After training:

```bat
.venv\Scripts\python.exe predict.py --text "Congratulations! You won a free prize. Click here now."
```

Optional model path:

```bat
.venv\Scripts\python.exe predict.py --text "See you tomorrow" --model-dir outputs/spam_classifier_scratch/best_model
```

Example output:

```python
{
  "label": "spam",
  "label_id": 1,
  "confidence": 0.98,
  "probabilities": {"ham": 0.02, "spam": 0.98}
}
```

Labels:

- `0` / `ham` — legitimate message  
- `1` / `spam` — spam / phishing-style message  

## Results & plots

Generate an HTML summary:

```bat
.venv\Scripts\python.exe generate_results_html.py
```

Or:

```bat
scripts\results.bat
```

This creates `results.html` from the saved training history and test report.

Optional metric plots:

```bat
.venv\Scripts\python.exe plot_metrics.py
```

Plots are written under `outputs/spam_classifier_scratch/plots/`.

## Configuration

Hyperparameters live in `config.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `dataset_name` | `LHRS-UM-FERI/MENTHOS-dataset-spam` | Hugging Face dataset |
| `batch_size` | `64` | Training batch size |
| `eval_batch_size` | `128` | Eval/test batch size |
| `num_epochs` | `15` | Training epochs |
| `learning_rate` | `3e-4` | AdamW learning rate |
| `weight_decay` | `0.01` | AdamW weight decay |
| `context_size` | `128` | Max sequence length |
| `model_dimension` | `128` | Embedding / model width |
| `number_of_blocks` | `2` | Encoder layers |
| `heads` | `4` | Attention heads |
| `feed_forward_dimension` | `512` | FFN hidden size |
| `dropout` | `0.1` | Dropout rate |
| `seed` | `42` | Reproducibility seed |

You can also limit samples with `max_train_samples`, `max_eval_samples`, and `max_test_samples` for quick experiments.

## Project structure

```text
.
├── config.py                 # Training / model hyperparameters
├── dataset.py                # Dataset loading, tokenizer, SpamDataset
├── model.py                  # Transformer building blocks + EncoderClassifier
├── train.py                  # Training loop and checkpointing
├── predict.py                # Single-message inference CLI
├── evaluate.py               # Metrics and test report helpers
├── generate_results_html.py  # HTML results page
├── plot_metrics.py           # Training / test metric plots
├── transformers.ipynb        # Notebook experiments
├── requirements-spam.txt     # Minimal runtime deps
├── requirements.txt          # Full environment (incl. Jupyter / CUDA)
├── sample_data/              # Sample CSV / custom test text
├── scripts/                  # Windows helpers (setup, train, predict, results)
└── run.bat                   # Launches the scripts menu
```

## Model overview

The classifier is an **encoder-only** Transformer:

1. Token embeddings + sinusoidal positional encoding  
2. Stacked encoder blocks (multi-head self-attention + feed-forward, with residual connections)  
3. Mean pooling over non-padding tokens  
4. Linear head → logits for `ham` / `spam`  

All weights are randomly initialized (Xavier) and trained end-to-end on the spam dataset.

## License

If this repository does not include a license file, add one before distributing the project publicly.
