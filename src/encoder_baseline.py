"""Fine-tune a small encoder classifier on Banking77 as a third comparison point.

Qwen2.5-7B + LoRA reached 88.7% accuracy (see train.py / evaluate.py) --
solid, but published state-of-the-art results on Banking77 come from
dedicated encoder classifiers rather than generative LLMs. A fine-tuned
ModernBERT-base (149M parameters, roughly 2% the size of Qwen2.5-7B)
reaches ~94% accuracy in published results.

This script reproduces that approach directly on this project's own
Banking77 split: full fine-tuning of a small encoder plus a linear
classification head, the standard method for supervised text
classification. Unlike a frozen-embeddings-plus-linear-probe approach,
this trains the encoder's weights too, which is what actually gets close
to the published numbers.

The point isn't just "which model wins" -- it's an honest answer to
whether a 7B generative model is even the right tool for a pure
classification task, compared to a purpose-built classifier an order of
magnitude smaller and cheaper to run.
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

MODEL_ID = "answerdotai/ModernBERT-base"
DEFAULT_OUTPUT_DIR = Path("checkpoints/modernbert-banking77")
RESULTS_PATH = Path("results/encoder_baseline.json")


def tokenize_dataset(dataset, tokenizer, max_length: int = 64):
    """Tokenize and rename the label column to what Trainer expects.

    Banking77's label column is named "label" (singular); the model's
    forward pass and Trainer's loss computation look for "labels"
    (plural) specifically, so this rename is required, not cosmetic.
    """

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
    """Fine-tune ModernBERT-base end-to-end on Banking77.

    Unlike the LoRA run in train.py, this trains all of the (much
    smaller) model's weights -- the standard approach for supervised
    text classification with encoder models, and what published Banking77
    benchmarks actually do.

    Uses the same training recipe (cosine schedule, warmup, batch size)
    as encoder_baseline_large.py, so any accuracy difference between the
    two is attributable to model size, not to differing hyperparameters.

    Returns (trainer, metrics) so callers can inspect results directly.
    """
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
            {
                "model": MODEL_ID,
                "accuracy": metrics["eval_accuracy"],
                "num_parameters": "149M",
            },
            f,
            indent=2,
        )
    print(f"Saved results to {RESULTS_PATH}")