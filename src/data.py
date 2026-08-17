"""Load and prepare the Banking77 intent classification dataset.

Banking77 contains ~13k customer support messages, each labeled with one
of 77 fine-grained banking intents (e.g. "card_not_working" vs
"declined_card_payment"). Unlike broad topic classification, these
categories overlap heavily, which is exactly why zero-shot prompting
alone tends to struggle here -- leaving real room for fine-tuning to
show a measurable difference.

We load the dataset via mteb/banking77 (a Parquet-format mirror of the
original PolyAI/banking77 dataset) rather than PolyAI/banking77
directly: as of `datasets` 4.x, Hugging Face dropped support for
datasets that ship a Python loading script, which PolyAI/banking77
still uses. The mteb mirror has identical content and splits, just in
the modern, script-free format.

This module is shared by baseline.py (zero-shot Qwen) and train.py
(LoRA fine-tuned Qwen), so both use the identical prompt format. That
matters: if the two scripts built prompts differently, any accuracy
difference between them could come from the prompt wording, not from
the fine-tuning itself.
"""

from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset

DATASET_NAME = "mteb/banking77"

SYSTEM_PROMPT = (
    "You are a banking customer support assistant. Classify the customer's "
    "message into exactly one of the given intent categories. Respond with "
    "the category name only, nothing else."
)


def load_raw_dataset() -> DatasetDict:
    """Load the raw Banking77 dataset (train/test splits)."""
    return load_dataset(DATASET_NAME)


def get_label_names(dataset: DatasetDict) -> list[str]:
    """Return the 77 intent category names, ordered by their label id.

    The mteb mirror stores label as a plain int column (not a ClassLabel
    feature), so we derive the ordered name list from the label/label_text
    pairs actually present in the data, rather than from feature metadata.
    """
    pairs = {(ex["label"], ex["label_text"]) for ex in dataset["train"]}
    return [name for _, name in sorted(pairs)]


def build_prompt(text: str, label_names: list[str]) -> str:
    """Build the user-facing prompt for a single banking query.

    Used identically by the baseline (zero-shot) and training pipeline,
    so both see the exact same question format.
    """
    categories = ", ".join(label_names)
    return (
        f"Categories: {categories}\n\n"
        f"Customer message: {text}\n\n"
        f"Category:"
    )


def format_for_training(example: dict, label_names: list[str]) -> dict:
    """Format one example as a chat-style sample for supervised fine-tuning.

    Produces a "messages" field in the format trl's SFTTrainer expects,
    with the correct intent name as the target assistant completion.
    """
    prompt = build_prompt(example["text"], label_names)
    completion = label_names[example["label"]]
    example["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    return example


def prepare_dataset() -> tuple[Dataset, Dataset, list[str]]:
    """Load Banking77 and return (train, test, label_names), ready to use."""
    raw = load_raw_dataset()
    label_names = get_label_names(raw)

    train = raw["train"].map(lambda ex: format_for_training(ex, label_names))
    test = raw["test"].map(lambda ex: format_for_training(ex, label_names))
    return train, test, label_names


if __name__ == "__main__":
    train, test, label_names = prepare_dataset()
    print(f"Train examples: {len(train)}")
    print(f"Test examples: {len(test)}")
    print(f"Number of intent categories: {len(label_names)}")
    print("\nExample training sample:")
    print(train[0]["messages"])