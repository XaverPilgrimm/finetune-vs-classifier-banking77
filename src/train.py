"""LoRA fine-tuning of Qwen2.5-7B-Instruct on Banking77.

Uses PEFT (LoRA adapters) rather than full fine-tuning: LoRA only trains
a small set of low-rank adapter weights instead of all 7B parameters,
which is what makes fine-tuning this model feasible on a single GPU in
a reasonable amount of time and disk space. The base model's weights
stay frozen; only the adapter is trained and saved.

This module reuses build_prompt() and SYSTEM_PROMPT from data.py, so
the fine-tuned model is trained on exactly the same prompt format the
zero-shot baseline was evaluated on -- otherwise a comparison between
baseline.py and this script would be confounded by prompt differences,
not just by the presence or absence of fine-tuning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from data import prepare_dataset

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_DIR = Path("checkpoints/qwen-banking77-lora")

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Attention + MLP projection layers -- the standard target set for
    # LoRA on Qwen/Llama-style architectures.
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)


def load_model_and_tokenizer(model_id: str = MODEL_ID):
    """Load the base model and tokenizer for training."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


def run_lora_finetune(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_steps: int | None = None,
):
    """Fine-tune Qwen2.5-7B-Instruct on Banking77 with LoRA.

    max_steps: if set, overrides num_train_epochs and stops after this
    many training steps regardless of dataset size. Useful for a quick
    smoke test (e.g. max_steps=5) to confirm the whole pipeline runs
    end-to-end -- model loads, LoRA applies, a training step succeeds,
    a checkpoint saves -- before committing to a full, slow run.

    Returns (trainer, output_dir) so callers (e.g. a notebook) can
    inspect trainer.state.log_history for the loss curve, or reload
    the adapter from output_dir afterwards.
    """
    train, _test, _label_names = prepare_dataset()
    model, tokenizer = load_model_and_tokenizer()

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=True,
        loss_type="nll",
        logging_steps=1 if max_steps is not None else 10,
        save_strategy="epoch" if max_steps is None else "no",
        report_to="none",
        max_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        peft_config=LORA_CONFIG,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return trainer, output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after N steps instead of full epochs (for a quick smoke test).",
    )
    args = parser.parse_args()

    trainer, output_dir = run_lora_finetune(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
    )
    print(f"\nAdapter saved to {output_dir}")