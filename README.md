# Fine-Tuning vs. Zero-Shot: Qwen2.5-7B on Banking77

Comparing Qwen2.5-7B-Instruct zero-shot against the same model fine-tuned
with LoRA, on a single controlled task: Banking77 intent classification.
Only one variable is isolated, fine-tuning, yes or no, on the same model,
the same prompt format, and the same answer-parsing logic.

## Task

[Banking77](https://huggingface.co/datasets/mteb/banking77): 9,993 training /
3,076 test customer support messages, labeled with one of 77 fine-grained,
overlapping banking intents (e.g. `card_not_working` vs.
`declined_card_payment`). Chosen over broader topic classification tasks
because those tend to already be near-saturated for a 7B-class instruction
model zero-shot, leaving little room for fine-tuning to demonstrate a
difference. Banking77's categories overlap by design, so zero-shot
prompting has genuine room to fail.

## Results

| Metric | Zero-shot | Fine-tuned |
|---|---|---|
| Accuracy | 62.8% | 88.7% |
| Correct | 1,932 | 2,729 |
| Wrong, valid category | 991 | 312 |
| Invalid (hallucinated) category | 153 | 35 |
| Unparseable | 0 | 0 |

Fine-tuning improved accuracy by nearly 26 percentage points. Notably, the
model's tendency to invent a plausible-sounding but nonexistent category
dropped by 77%, a larger relative improvement than the drop in genuine
wrong-but-valid mistakes (68%), suggesting fine-tuning disproportionately
helped the model stay within the allowed label set. See the notebook's
Conclusion section for the full discussion, including likely reasons the
model still falls short of perfect accuracy.

## Stack

- **Qwen2.5-7B-Instruct** (base model, Apache 2.0)
- **PEFT (LoRA)** for parameter-efficient fine-tuning
- **trl (SFTTrainer)** for the training loop
- **Hugging Face `datasets` / `transformers`**
- Trained on 4x NVIDIA H100 NVL (95 GB)

## Workflow

Dataset (`src/data.py`) → Zero-shot baseline (`src/baseline.py`) → LoRA
fine-tuning (`src/train.py`) → Evaluation with a four-category error
breakdown (`src/evaluate.py`) → Comparison

## Setup

1. Clone the repo
2. This project assumes a CUDA-enabled PyTorch environment is already
   available (developed against `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`);
   install the remaining dependencies: `pip install -r requirements.txt`
3. Run `notebook.ipynb` top to bottom

For a quick pipeline check before committing to a full run: