"""Shared helpers: cleaning, data loading, metrics."""
import json
import re
from pathlib import Path

from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

MODELS_DIR = Path("models")
METRICS_FILE = MODELS_DIR / "metrics.json"
LABELS = ["Negative", "Positive"]


def clean_text(text: str) -> str:
    """Remove HTML line breaks/tags and extra whitespace (keep case & punctuation:
    BERT's tokenizer handles those, and they carry sentiment signal)."""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_imdb(train_samples=None, test_samples=2000, seed=42):
    from datasets import load_dataset
    ds = load_dataset("imdb")
    train = ds["train"].shuffle(seed=seed)
    test = ds["test"].shuffle(seed=seed)
    if train_samples:
        train = train.select(range(min(train_samples, len(train))))
    if test_samples:
        test = test.select(range(min(test_samples, len(test))))
    return ([clean_text(t) for t in train["text"]], list(train["label"]),
            [clean_text(t) for t in test["text"]], list(test["label"]))


def compute_metrics(y_true, y_pred) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p,
            "recall": r, "f1": f1,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()}


def save_metrics(name: str, metrics: dict, extra: dict | None = None):
    MODELS_DIR.mkdir(exist_ok=True)
    data = json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else {}
    data[name] = {**metrics, **(extra or {})}
    METRICS_FILE.write_text(json.dumps(data, indent=2))
