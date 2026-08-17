"""Evaluate and compare zero-shot vs. fine-tuned Qwen2.5-7B on Banking77.

Rather than re-running generation, this module analyzes the raw
prediction records already saved by baseline.py (and produced by
run_finetuned_eval below) -- each record already contains the model's
raw text output, so we can classify outcomes after the fact without
paying for inference again.

Splitting "wrong" into three distinct outcomes matters here because we
found, while running the baseline, that the model doesn't just guess
incorrectly -- it sometimes invents a plausible-sounding category that
isn't one of the 77 allowed labels at all (e.g. "getting_physical_card"
vs. the real "get_physical_card"). Lumping that together with a
genuine wrong-but-valid guess would hide a real, distinct failure mode
that fine-tuning might specifically fix: staying within the allowed
label set, not just picking the right one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline import MODEL_ID, match_label, predict_one
from data import prepare_dataset

RESULTS_DIR = Path("results")
CATEGORIES = ["correct", "wrong_valid", "invalid_category", "unparseable"]


def classify_prediction(
    raw_output: str, predicted_label: str | None, true_label: str
) -> str:
    """Classify one prediction into one of four distinct outcomes.

    - "correct": predicted the right label.
    - "wrong_valid": predicted a real category from the label set, just
      not the right one -- a genuine classification mistake.
    - "invalid_category": the model produced category-like text that
      isn't in the allowed label set at all (a made-up label).
    - "unparseable": no category-like answer could be extracted at all
      (e.g. a refusal, or an empty/garbled response).
    """
    if predicted_label is not None:
        return "correct" if predicted_label == true_label else "wrong_valid"

    first_line = raw_output.strip().split("\n")[0].strip().strip(".")
    return "invalid_category" if first_line else "unparseable"


def analyze_results(results: dict) -> dict:
    """Add a four-category error breakdown to a raw results dict.

    Works on the output of either baseline.py's run_baseline() or
    run_finetuned_eval() below -- both save the same "predictions"
    structure (text, true_label, predicted_label, raw_output).
    """
    counts = {category: 0 for category in CATEGORIES}
    for prediction in results["predictions"]:
        category = classify_prediction(
            prediction["raw_output"],
            prediction["predicted_label"],
            prediction["true_label"],
        )
        prediction["category"] = category
        counts[category] += 1

    results["counts"] = counts
    results["accuracy"] = counts["correct"] / len(results["predictions"])
    return results


def load_finetuned_model(adapter_dir: Path, base_model_id: str = MODEL_ID):
    """Load the base model with the LoRA adapter applied on top."""
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    return model, tokenizer


def run_finetuned_eval(adapter_dir: Path, limit: int | None = None) -> dict:
    """Run the fine-tuned model over the test set, same format as baseline.py.

    Reuses predict_one() and match_label() from baseline.py so
    generation and answer-parsing are identical between the zero-shot
    and fine-tuned runs -- only the model weights differ.
    """
    _, test, label_names = prepare_dataset()
    if limit is not None:
        test = test.select(range(limit))

    model, tokenizer = load_finetuned_model(adapter_dir)

    predictions = []
    for i, example in enumerate(test):
        raw_output = predict_one(model, tokenizer, example["text"], label_names)
        predicted = match_label(raw_output, label_names)
        predictions.append(
            {
                "text": example["text"],
                "true_label": label_names[example["label"]],
                "predicted_label": predicted,
                "raw_output": raw_output,
            }
        )
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(test)} evaluated")

    results = {
        "model": f"{MODEL_ID} + LoRA ({adapter_dir})",
        "num_examples": len(test),
        "predictions": predictions,
    }
    return analyze_results(results)


def compare_results(baseline_path: Path, finetuned_path: Path) -> None:
    """Print a side-by-side comparison table of baseline vs. fine-tuned results."""
    with open(baseline_path) as f:
        baseline = analyze_results(json.load(f))
    with open(finetuned_path) as f:
        finetuned = analyze_results(json.load(f))

    print(f"{'Metric':<20}{'Baseline':>12}{'Fine-tuned':>14}")
    print("-" * 46)
    print(f"{'Accuracy':<20}{baseline['accuracy']:>12.3f}{finetuned['accuracy']:>14.3f}")
    for category in CATEGORIES:
        b = baseline["counts"][category]
        ft = finetuned["counts"][category]
        print(f"{category:<20}{b:>12}{ft:>14}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-dir", type=Path, default=Path("checkpoints/qwen-banking77-lora")
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "finetuned.json")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="After evaluating, print a comparison against results/baseline.json.",
    )
    args = parser.parse_args()

    results = run_finetuned_eval(args.adapter_dir, limit=args.limit)

    print(f"\nAccuracy: {results['accuracy']:.3f}")
    print(f"Breakdown: {results['counts']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output}")

    if args.compare:
        print()
        compare_results(RESULTS_DIR / "baseline.json", args.output)