# LoRA Fine-Tuning vs. Purpose-Built Classifiers on Banking77

Three approaches to a single, controlled classification task: Qwen2.5-7B-Instruct
zero-shot, the same model fine-tuned with LoRA, and small fine-tuned encoder
classifiers (ModernBERT-base and -large) purpose-built for classification
rather than text generation.

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

| Model | Params | Accuracy | Invalid Category |
|---|---|---|---|
| Qwen2.5-7B zero-shot | 7B | 62.8% | 153 |
| Qwen2.5-7B + LoRA | 7B | 88.7% | 35 |
| ModernBERT-base fine-tuned | 149M | 93.5% | N/A |
| ModernBERT-large fine-tuned | 395M | 94.2% | N/A |

*"Invalid Category" (a hallucinated, non-existent label) is only possible for
generative models. ModernBERT selects directly from the 77 valid labels, so
this failure mode cannot occur by construction.*

LoRA fine-tuning improved Qwen's accuracy by nearly 26 percentage points, and
disproportionately reduced hallucinated categories (77% drop) versus
genuine wrong-but-valid mistakes (68% drop). But a purpose-built classifier
beats both Qwen variants: ModernBERT-large matches published state-of-the-art
(~94-95%) at 5.6% of Qwen's parameter count, training in minutes rather than
an hour. See the notebook's Conclusion for the full discussion.

## Stack

- **Qwen2.5-7B-Instruct** (Apache 2.0) + **PEFT (LoRA)** + **trl (SFTTrainer)**
- **ModernBERT-base / -large** (Apache 2.0), full fine-tuning via `transformers.Trainer`
- **Hugging Face `datasets` / `transformers`**
- Trained on 4x NVIDIA H100 NVL (95 GB)

## Workflow

Dataset (`src/data.py`) → Qwen zero-shot baseline (`src/baseline.py`) → LoRA
fine-tuning (`src/train.py`) → Qwen evaluation with a four-category error
breakdown (`src/evaluate.py`) → Encoder fine-tuning (`src/encoder_baseline.py`,
`src/encoder_baseline_large.py`) → Comparison

## Setup

1. Clone the repo
2. This project assumes a CUDA-enabled PyTorch environment is already
   available (developed against `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`);
   install the remaining dependencies: `pip install -r requirements.txt`
3. Run `notebook.ipynb` top to bottom

For a quick pipeline check before committing to a full run:
```
python src/baseline.py --limit 20
python src/train.py --max-steps 5
python src/encoder_baseline.py --epochs 1
```

## Project structure

- `src/data.py`, dataset loading and shared prompt formatting
- `src/baseline.py`, zero-shot evaluation and answer-parsing logic
- `src/train.py`, LoRA fine-tuning
- `src/evaluate.py`, fine-tuned model evaluation and baseline/fine-tuned comparison
- `src/encoder_baseline.py`, `src/encoder_baseline_large.py`, ModernBERT fine-tuning
- `notebook.ipynb`, end-to-end walkthrough with results and discussion

## Notes

- `checkpoints/` is gitignored (model weights, too large for the repo);
  `results/` is committed, it contains the actual prediction records, loss
  curve, and per-model results behind the numbers above