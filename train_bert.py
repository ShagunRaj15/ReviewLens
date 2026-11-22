"""Fine-tune a pretrained BERT model for sentiment (IMDb).

Use a GPU (e.g. free Google Colab T4). Example:
    python train_bert.py --train_samples 5000 --epochs 2
"""
import argparse
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from common import MODELS_DIR, compute_metrics, load_imdb, save_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="bert-base-uncased")  # or distilbert-base-uncased
ap.add_argument("--train_samples", type=int, default=5000)
ap.add_argument("--test_samples", type=int, default=2000)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-5)
args = ap.parse_args()


class TextDataset(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(texts, truncation=True, max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


X_tr, y_tr, X_te, y_te = load_imdb(args.train_samples, args.test_samples)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=2, id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1})

train_ds = TextDataset(X_tr, y_tr, tok, args.max_len)
test_ds = TextDataset(X_te, y_te, tok, args.max_len)

trainer = Trainer(
    model=model, train_dataset=train_ds, tokenizer=tok,
    args=TrainingArguments(
        output_dir="models/_runs", num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32, learning_rate=args.lr,
        weight_decay=0.01, warmup_ratio=0.1, logging_steps=50,
        save_strategy="no", report_to="none",
        fp16=torch.cuda.is_available()))

t0 = time.time()
trainer.train()
train_time = time.time() - t0

preds = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
metrics = compute_metrics(y_te, preds)
print({k: round(v, 4) for k, v in metrics.items() if k != "confusion_matrix"})

trainer.save_model(str(MODELS_DIR / "bert"))
tok.save_pretrained(str(MODELS_DIR / "bert"))
save_metrics(f"Fine-tuned {args.model}", metrics,
             {"train_samples": len(X_tr), "test_samples": len(X_te),
              "epochs": args.epochs, "train_seconds": round(train_time, 1)})
print("Saved model to models/bert")
