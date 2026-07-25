import json
from datetime import datetime
from pathlib import Path

from config import get_config


def pct(value: float) -> str:
  return f"{value * 100:.2f}%"


def load_json(path: Path) -> dict | None:
  if not path.exists():
    return None
  return json.loads(path.read_text(encoding="utf-8"))


def get_training_validation(trainer_state_path: Path) -> dict | None:
  state = load_json(trainer_state_path)
  if not state:
    return None

  eval_entries = [
    entry for entry in state.get("log_history", [])
    if "eval_f1" in entry
  ]
  if not eval_entries:
    return None

  best = max(eval_entries, key=lambda entry: entry["eval_f1"])
  return {
    "epoch": best.get("epoch"),
    "accuracy": best.get("eval_accuracy"),
    "precision": best.get("eval_precision"),
    "recall": best.get("eval_recall"),
    "f1": best.get("eval_f1"),
    "loss": best.get("eval_loss"),
    "completed_epochs": state.get("epoch"),
    "planned_epochs": state.get("num_train_epochs"),
  }


def metric_row(label: str, value: float | None) -> str:
  if value is None:
    return f"<tr><td>{label}</td><td class='muted'>—</td></tr>"
  return f"<tr><td>{label}</td><td><strong>{pct(value)}</strong></td></tr>"


def class_table(report: dict, class_name: str) -> str:
  data = report[class_name]
  return f"""
    <tr>
      <td>{class_name.upper()}</td>
      <td>{pct(data['precision'])}</td>
      <td>{pct(data['recall'])}</td>
      <td><strong>{pct(data['f1-score'])}</strong></td>
      <td>{int(data['support']):,}</td>
    </tr>
  """


def bar(value: float, label: str) -> str:
  width = max(0, min(100, value * 100))
  return f"""
    <div class="bar-row">
      <div class="bar-label">{label}</div>
      <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>
      <div class="bar-value">{pct(value)}</div>
    </div>
  """


def generate_html(output_path: Path) -> None:
  config = get_config()
  base = Path(config["output_dir"])

  test_report = load_json(base / "test_classification_report.json")
  if not test_report:
    raise FileNotFoundError(
      f"Nije pronadjen test izvestaj: {base / 'test_classification_report.json'}"
    )

  trainer_validation = get_training_validation(base / "checkpoint-1906" / "trainer_state.json")
  if trainer_validation is None:
    for checkpoint in sorted(base.glob("checkpoint-*")):
      trainer_validation = get_training_validation(checkpoint / "trainer_state.json")
      if trainer_validation:
        break

  generated_at = datetime.now().strftime("%d.%m.%Y. %H:%M")
  accuracy = test_report["accuracy"]

  html = f"""<!DOCTYPE html>
<html lang="sr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SMS Spam Classifier — Rezultati</title>
  <style>
    :root {{
      --bg: #0f1419;
      --card: #1a2332;
      --border: #2d3a4f;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --accent: #3b82f6;
      --accent-2: #22c55e;
      --ham: #38bdf8;
      --spam: #f97316;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: linear-gradient(160deg, #0b1020 0%, #121a2b 45%, #0f1419 100%);
      color: var(--text);
      line-height: 1.5;
      padding: 2rem 1rem 3rem;
    }}
    .container {{ max-width: 980px; margin: 0 auto; }}
    h1 {{ margin: 0 0 0.35rem; font-size: 1.9rem; }}
    .subtitle {{ color: var(--muted); margin-bottom: 1.5rem; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.1rem 1.25rem;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }}
    .card h2 {{
      margin: 0 0 0.85rem;
      font-size: 1rem;
      color: var(--muted);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .hero {{
      text-align: center;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: radial-gradient(circle at top, rgba(59,130,246,0.18), transparent 55%), var(--card);
    }}
    .hero .score {{
      font-size: 3rem;
      font-weight: 700;
      color: var(--accent-2);
      line-height: 1.1;
    }}
    .hero .score-label {{ color: var(--muted); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      padding: 0.65rem 0.5rem;
      border-bottom: 1px solid var(--border);
      text-align: left;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    tr:last-child td {{ border-bottom: none; }}
    .tag {{
      display: inline-block;
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
    }}
    .tag-ham {{ background: rgba(56,189,248,0.15); color: var(--ham); }}
    .tag-spam {{ background: rgba(249,115,22,0.15); color: var(--spam); }}
    .meta {{ color: var(--muted); font-size: 0.9rem; }}
    .meta dt {{ font-weight: 600; margin-top: 0.55rem; }}
    .meta dd {{ margin: 0.15rem 0 0; }}
    .bar-row {{
      display: grid;
      grid-template-columns: 90px 1fr 70px;
      gap: 0.75rem;
      align-items: center;
      margin-bottom: 0.7rem;
    }}
    .bar-label {{ color: var(--muted); font-size: 0.9rem; }}
    .bar-track {{
      height: 10px;
      background: #111827;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      border-radius: 999px;
    }}
    .bar-value {{ text-align: right; font-weight: 600; }}
    .muted {{ color: var(--muted); }}
    .footer {{
      margin-top: 1.5rem;
      text-align: center;
      color: var(--muted);
      font-size: 0.85rem;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Detekcija spam SMS poruka</h1>
    <p class="subtitle">Pregled rezultata treniranja — DistilBERT + MENTHOS dataset</p>

    <div class="hero card">
      <div class="score">{pct(accuracy)}</div>
      <div class="score-label">Test accuracy (13.128 poruka)</div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Test — F1</h2>
        {bar(test_report['ham']['f1-score'], '<span class="tag tag-ham">HAM</span>')}
        {bar(test_report['spam']['f1-score'], '<span class="tag tag-spam">SPAM</span>')}
      </div>
      <div class="card">
        <h2>Test — ukupno</h2>
        <table>
          {metric_row('Accuracy', test_report['accuracy'])}
          {metric_row('Macro F1', test_report['macro avg']['f1-score'])}
          {metric_row('Weighted F1', test_report['weighted avg']['f1-score'])}
        </table>
      </div>
      <div class="card">
        <h2>Model</h2>
        <dl class="meta">
          <dt>Arhitektura</dt><dd>{config['model_name']}</dd>
          <dt>Dataset</dt><dd>{config['dataset_name']}</dd>
          <dt>Max dužina</dt><dd>{config['max_length']} tokena</dd>
          <dt>Epoha (plan)</dt><dd>{config['num_epochs']}</dd>
        </dl>
      </div>
    </div>

    <div class="card" style="margin-bottom: 1rem;">
      <h2>Detaljne metrike — test skup</h2>
      <table>
        <thead>
          <tr>
            <th>Klasa</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>Support</th>
          </tr>
        </thead>
        <tbody>
          {class_table(test_report, 'ham')}
          {class_table(test_report, 'spam')}
          <tr class="muted">
            <td>MACRO AVG</td>
            <td>{pct(test_report['macro avg']['precision'])}</td>
            <td>{pct(test_report['macro avg']['recall'])}</td>
            <td><strong>{pct(test_report['macro avg']['f1-score'])}</strong></td>
            <td>{int(test_report['macro avg']['support']):,}</td>
          </tr>
        </tbody>
      </table>
    </div>
"""

  if trainer_validation:
    html += f"""
    <div class="card">
      <h2>Validation tokom treninga (najbolja epoha)</h2>
      <table>
        {metric_row('Accuracy', trainer_validation.get('accuracy'))}
        {metric_row('Precision', trainer_validation.get('precision'))}
        {metric_row('Recall', trainer_validation.get('recall'))}
        {metric_row('F1', trainer_validation.get('f1'))}
        {metric_row('Loss', trainer_validation.get('loss'))}
      </table>
      <p class="meta" style="margin-top: 0.75rem;">
        Završeno epoha: {trainer_validation.get('completed_epochs')} / {trainer_validation.get('planned_epochs')}
      </p>
    </div>
"""

  html += f"""
    <p class="footer">Generisano: {generated_at} · Fajl: {output_path.name}</p>
  </div>
</body>
</html>
"""

  output_path.write_text(html, encoding="utf-8")
  print(f"Sacuvano: {output_path.resolve()}")


if __name__ == "__main__":
  generate_html(Path("results.html"))
