"""Experimental: ModernBERT-large + cosine LR schedule on Banking77.

One-off attempt to push accuracy closer to published state-of-the-art
(~94-95%). Larger encoder (395M vs. 149M params) and a cosine decay
schedule with warmup instead of linear decay with no warmup. Kept
separate from encoder_baseline.py so it can be dropped without touching
the already-working ModernBERT-base result (93.5% accuracy).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data import prepare_dataset

MODEL_ID = "answerdotai/ModernBERT-large"
DEFAULT_OUTPUT_DIR = Path("checkpoints/modernbert-large-banking77")
RESULTS_PATH = Path("results/encoder_baseline_large.json")


def tokenize_dataset(dataset, tokenizer, max_length: int = 64):
    def _tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(_tokenize, batched=True)
    return tokenized.rename_column("label", "labels")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}


def run_encoder_finetune(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    num_train_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    lr_scheduler_type: str = "cosine",
    warmup_steps: float = 0.1,
):
    train, test, label_names = prepare_dataset()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(label_names)
    )

    train_tok = tokenize_dataset(train, tokenizer)
    test_tok = tokenize_dataset(test, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    return trainer, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--warmup-steps", type=float, default=0.1)
    args = parser.parse_args()

    trainer, metrics = run_encoder_finetune(
        num_train_epochs=args.epochs,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
    )
    print(f"\nFinal accuracy: {metrics['eval_accuracy']:.3f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {"model": MODEL_ID, "accuracy": metrics["eval_accuracy"], "num_parameters": "395M"},
            f,
            indent=2,
        )
    print(f"Saved results to {RESULTS_PATH}")