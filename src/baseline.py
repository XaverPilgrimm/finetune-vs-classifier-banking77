"""Zero-shot baseline: Qwen2.5-7B-Instruct on Banking77, no fine-tuning.

This is the control point for the whole experiment. Without it, an
accuracy improvement from the fine-tuned model would be uninterpretable
-- we wouldn't know whether it came from the fine-tuning itself, or
whether the base model could already do most of the work through
prompting alone. This script answers "how good is Qwen out of the box"
so train.py's fine-tuned result has something meaningful to compare
against.

It reuses build_prompt() from data.py so the baseline sees the exact
same question format as the fine-tuned model will.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import build_prompt, prepare_dataset

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
RESULTS_PATH = Path("results/baseline.json")


def load_model(model_id: str = MODEL_ID):
    """Load the base model and tokenizer, no adapters, no fine-tuning."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def predict_one(model, tokenizer, text: str, label_names: list[str]) -> str:
    """Run the model on a single banking query, return its raw text output."""
    prompt = build_prompt(text, label_names)
    messages = [{"role": "user", "content": prompt}]
    chat_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def match_label(raw_output: str, label_names: list[str]) -> str | None:
    """Map the model's free-text output back to a known label name.

    The model is instructed to answer with the category name only, but
    generation isn't perfectly constrained -- it may add punctuation or
    wrap the answer in a sentence. We try an exact match first, then
    fall back to checking whether a known label appears as a substring.
    """
    cleaned = raw_output.strip().strip(".").strip()
    if cleaned in label_names:
        return cleaned

    for name in label_names:
        if name in raw_output:
            return name

    return None


def run_baseline(limit: int | None = None) -> dict:
    """Run the zero-shot baseline over the Banking77 test set.

    limit: if set, only evaluate the first N test examples (useful for
    a quick smoke test before committing to a full, slower run).
    """
    _, test, label_names = prepare_dataset()
    if limit is not None:
        test = test.select(range(limit))

    model, tokenizer = load_model()

    correct = 0
    unparseable = 0
    predictions = []

    for i, example in enumerate(test):
        raw_output = predict_one(model, tokenizer, example["text"], label_names)
        predicted = match_label(raw_output, label_names)
        true_label = label_names[example["label"]]

        if predicted is None:
            unparseable += 1
        elif predicted == true_label:
            correct += 1

        predictions.append(
            {
                "text": example["text"],
                "true_label": true_label,
                "predicted_label": predicted,
                "raw_output": raw_output,
            }
        )

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(test)} evaluated")

    accuracy = correct / len(test)
    results = {
        "model": MODEL_ID,
        "num_examples": len(test),
        "correct": correct,
        "unparseable": unparseable,
        "accuracy": accuracy,
        "predictions": predictions,
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N test examples (for a quick smoke test).",
    )
    args = parser.parse_args()

    results = run_baseline(limit=args.limit)

    print(f"\nAccuracy: {results['accuracy']:.3f}")
    print(f"Correct: {results['correct']}/{results['num_examples']}")
    print(f"Unparseable outputs: {results['unparseable']}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {RESULTS_PATH}")