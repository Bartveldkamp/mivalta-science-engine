# MiValta Josi LLM Training Review

**Date:** 2026-02-09
**Scope:** Full review of the LLM fine-tuning pipeline for the Josi coaching assistant
**Codebase:** `mivalta-science-engine`

---

## 1. Executive Summary

MiValta fine-tunes open-source LLMs using LoRA to create "Josi", an AI coaching assistant that explains athletic training decisions based on sports physiology. The pipeline covers dataset generation, fine-tuning, validation, and GGUF export for mobile deployment.

**Deployed model:** The actual production model is **SmolLM2** (HuggingFaceTB) — exported as `josi-smollm2-merged-v2-q4_k_m.gguf`. However, the training scripts in the repository only reference Mistral-7B-Instruct-v0.3 and Ministral-3B. There is no SmolLM2 fine-tuning script in the codebase, which means the script used to train the deployed model is either missing from version control or maintained elsewhere.

**Overall assessment:** The project has a well-structured knowledge base and a thoughtful approach to persona-driven coaching. However, there are several areas in the training pipeline that warrant attention — most critically, the **mismatch between the codebase and the actual deployed model**. Beyond that, dataset quality concerns, training configuration choices, and validation coverage gaps need attention.

---

## 2. Architecture Overview

```
Knowledge Cards (18 .md files)
        |
        v
Generated Python Modules (context.py, tables.py)
        |
        v
Dataset Generators (generate_dataset.py, generate_philosophy_enhanced.py)
        |
        v
Training Data (~26K examples in JSONL)
        |
        v
LoRA Fine-Tuning (finetune_mistral.py / finetune_ministral3b.py)
        |
        v
Validation (validate_josi.py — 50 prompts, 10 categories)
        |
        v
GGUF Export (export_gguf.py — q4_k_m for mobile)
```

---

## 3. Dataset Analysis

### 3.1 Dataset Composition

| Dataset File | Examples | Format |
|---|---|---|
| `train_chat.jsonl` | 4,543 | Chat (system/user/assistant) |
| `val_chat.jsonl` | 505 | Chat |
| `train.jsonl` | 4,543 | Instruction/response |
| `val.jsonl` | 505 | Instruction/response |
| `gold_combined.jsonl` | 678 | Chat |
| `philosophy_*.jsonl` | ~900 | Chat |
| `round3-7_train.jsonl` | ~10,000 | Chat |

### 3.2 Findings

**F-DS1: Heavy template-driven generation, low diversity.**
The core dataset generator (`generate_dataset.py`) uses rigid templates with string interpolation. Every response for a given persona + readiness + zone combination is deterministic (or selected from a small pool of 2-4 variants). This means the model will see near-identical phrasing thousands of times with only the variable slots changed. This risks the model memorizing templates rather than learning to reason about coaching.

**F-DS2: Extreme duplication via repetition loops.**
Natural zone Q&A is generated with `for _ in range(3)` (3x repetition per variant per zone per persona). Natural conversations use `for _ in range(8)` (8x repetition). Simple "why" questions use `for _ in range(25)` per context per persona. These loops produce byte-identical duplicates in the training set. Duplicates inflate the dataset without adding information, and cause the model to overfit to those specific phrasings.

**F-DS3: Dataset imbalance across task types.**
The natural conversation and simple "why" examples are massively overrepresented due to the repetition loops, while critical behaviors like I6 refusals (model refusing to modify plans) have far fewer examples. The I6 refusal set has ~100 examples (25 violations x 4 personas) compared to potentially thousands of natural conversation duplicates. This imbalance means the safety-critical I6 behavior is underrepresented.

**F-DS4: Train/validation split happens after shuffling all examples.**
The 90/10 split is applied after shuffling the full generated dataset. Because many examples share nearly identical templates (differing only in a zone name or readiness level), it is likely that the validation set contains examples that are near-duplicates of training examples. This means validation loss will underestimate true generalization error.

**F-DS5: Inconsistent dataset usage across training scripts.**
`finetune_mistral.py` uses `train_chat.jsonl` (4,543 examples), while `finetune_ministral3b.py` uses `gold_combined.jsonl` (678 examples). The relationship between these datasets and the larger `round3-7_train.jsonl` (~10,000 examples) is unclear. There is no documentation indicating which dataset should be used for which training run, or how the "rounds" relate to each other.

**F-DS6: System prompt varies by persona but validation doesn't test persona consistency.**
Each chat example includes a system prompt like `"You are Josi... Style: warm, professional, supportive."` The system prompt changes per persona, but the validation script uses a single uniform prompt format without persona specification. This means persona adherence is never tested.

---

## 4. Training Configuration Review

### 4.1 Mistral 7B Fine-Tuning (`finetune_mistral.py`)

| Parameter | Value | Assessment |
|---|---|---|
| Base model | Mistral-7B-Instruct-v0.3 | Reasonable choice |
| LoRA rank | 64 | High — effective for domain adaptation |
| LoRA alpha | 128 | Alpha = 2x rank — standard practice |
| LoRA dropout | 0.05 | Standard |
| Target modules | All attention + MLP projections (7 modules) | Aggressive — all linear layers are adapted |
| Learning rate | 2e-4 | On the high end for LoRA fine-tuning |
| Effective batch size | 16 (4 x 4 accumulation) | Reasonable |
| Epochs | 3 | Standard for LoRA |
| Max seq length | 1024 | May truncate longer conversations |
| Quantization | 4-bit NF4 + double quant | Good for memory efficiency |
| Optimizer | paged_adamw_8bit | Good for memory efficiency |
| LR scheduler | Cosine | Standard |
| Max grad norm | 0.3 | Aggressive clipping — conservative |

### 4.2 Findings

**F-TR1: `pad_token = eos_token` can cause generation issues.**
Setting the pad token to the EOS token (`finetune_mistral.py:110`) is a common workaround, but it means the model cannot distinguish between "end of sequence" and "padding". During inference, this can cause the model to stop generating prematurely or, conversely, to ignore the EOS signal. A dedicated pad token or using `unk_token` as pad would be safer.

**F-TR2: Manual chat template formatting instead of using the tokenizer's built-in template.**
`format_chat_message()` (`finetune_mistral.py:138-158`) manually constructs the Mistral `[INST]...[/INST]` format with string concatenation. This is fragile and prone to subtle formatting errors (e.g., missing `<s>` tags for multi-turn, incorrect whitespace). The `transformers` library provides `tokenizer.apply_chat_template()` which handles this correctly for each model. The current implementation also handles the system message incorrectly — Mistral-Instruct v0.3 has a specific `[SYSTEM_PROMPT]` format, but the code concatenates the system message into the first `[INST]` block.

**F-TR3: The Ministral 3B script uses `Mistral3ForConditionalGeneration`.**
`finetune_ministral3b.py:9,29` imports and uses `Mistral3ForConditionalGeneration`, which is a multimodal model class (vision + language). For a text-only fine-tuning task, `AutoModelForCausalLM` should be used instead. This may load unnecessary vision components or fail entirely on newer transformers versions.

**F-TR4: Ministral 3B script lacks validation split.**
`finetune_ministral3b.py` trains on `gold_combined.jsonl` without any validation set. There is no eval strategy configured, so there is no way to detect overfitting during training.

**F-TR5: Ministral 3B drops the system message entirely.**
The `format_example()` function in `finetune_ministral3b.py:51-62` only extracts the user and assistant messages, discarding the system message. Since the system message defines Josi's persona and behavioral constraints (including the critical "never prescribe" instruction), this means the 3B model is trained without the core behavioral guardrail.

**F-TR6: No early stopping configured.**
Neither training script uses early stopping. With 3 epochs over a dataset that contains many near-duplicates, overfitting is a real risk. The model may memorize template responses rather than generalizing.

**F-TR7: LoRA applied to all linear layers is aggressive for the 7B model.**
Targeting all 7 projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with rank 64 creates a large number of trainable parameters. This is appropriate if the domain is very different from the base model's training data, but for coaching conversations (which are close to natural language), a more conservative configuration (e.g., rank 16-32, attention layers only) might generalize better and reduce overfitting risk.

**F-TR8: `max_seq_length = 1024` but SFTTrainer field name may be wrong.**
The `SFTConfig` is set up correctly with training arguments, but the `max_seq_length` variable defined at line 67 is never actually passed to `SFTConfig`. The field for SFTConfig is `max_seq_length` but it is not included in the config at lines 199-216. This means the default max length (likely the model's full context window) may be used, or truncation may not happen as expected.

**F-TR9: `report_to="none"` — no experiment tracking.**
Both scripts disable all experiment tracking. For a project with multiple training rounds (7+), dataset variations, and two model sizes, there is no record of which hyperparameters or datasets produced which results. Enabling Weights & Biases (already listed as optional in `requirements.txt`) would help track training curves, compare runs, and diagnose issues.

---

## 5. Validation & Evaluation Review

### 5.1 Current Validation (`validate_josi.py`)

The validation script runs 50 prompts across 10 categories through `llama-cli` and evaluates responses against:
- Forbidden words (GATC internals like "algorithm", "viterbi", "hmm")
- Jargon (technical terms like "periodization", "mesocycle")
- Warmth score (keyword-based, 1-5)
- Question-asking rate
- Pushback on unrealistic goals
- Word count (20-300 words)

### 5.2 Findings

**F-VL1: Validation uses `llama-cli` output parsing, which is brittle.**
The `run_prompt()` function (`validate_josi.py:170-231`) runs `llama-cli` as a subprocess and parses the mixed stdout/stderr output by looking for `[/INST]` markers and filtering noise patterns. This is fragile — different llama.cpp versions may change output format, and the noise filters may miss new patterns or accidentally strip valid response content.

**F-VL2: Warmth scoring is simplistic keyword matching.**
The warmth score (`validate_josi.py:144-149`) counts occurrences of warm indicators ("I hear", "I get it") vs cold indicators ("studies show that", "data suggests"). This conflates persona style with quality. The "science_nerd" persona is designed to use analytical language that would score as "cold" by this metric, yet that is the intended behavior for that persona.

**F-VL3: I6 contract compliance is not directly tested.**
The validation script does not include prompts that attempt to get the model to prescribe or modify training. The "unrealistic_goals" category tests pushback on unsafe goals, but the I6 contract (model must refuse to modify plans) is a different behavior. Examples like "Change my workout to intervals" or "Add more Z5 to my week" are missing from validation, despite being a core training objective.

**F-VL4: No multi-turn evaluation.**
All 50 test prompts are single-turn. The training data includes follow-up conversations (thanks, disagreement, more_info), but the model's ability to maintain context, persona, and I6 compliance across multiple turns is never tested.

**F-VL5: 50 prompts is a small evaluation set.**
With 10 categories and 5 prompts each, there is limited statistical power to detect regressions. A single prompt failing or passing can swing a category by 20%.

**F-VL6: No automated regression testing between model versions.**
There is no mechanism to compare validation results across training runs. With 7+ training rounds, the project would benefit from storing validation results and comparing pass rates, warmth scores, and failure modes across versions.

---

## 6. Deployment Pipeline Review

### 6.1 GGUF Export (`export_gguf.py`)

**F-DP1: No post-quantization validation.**
The export script converts and quantizes the model but does not run any validation on the quantized output. Quantization (especially at q3_k_m or q2_k) can degrade response quality significantly. The validation suite should be run on the GGUF output, not just the full-precision model.

**F-DP2: Recommended quantization (q4_k_m) is reasonable.**
The default recommendation of q4_k_m for mobile is sound — it provides a good quality/size tradeoff for a 7B model at ~4GB.

---

## 7. Knowledge Base Review

### 7.1 Strengths

- **Well-structured knowledge cards:** 18 markdown files covering zones, periodization, recovery, load monitoring, modifiers, and sport-specific details.
- **Research-backed:** The zone physiology card cites specific research (Seiler 2010, Olympiatoppen I-Scale, Critical Power Model).
- **Clear separation of concerns:** Cards define boundaries (e.g., "Zone Physiology defines meaning, not planning").
- **Compiled to Python modules:** The `context.py` and `tables.py` generated modules make the knowledge programmatically accessible.

### 7.2 Findings

**F-KB1: Knowledge cards define 9 zones (R, Z1-Z8) but training data only uses 7 (R, Z1-Z6).**
The `ZONES` dict in `generate_dataset.py` omits Z7 (Maximal Anaerobic) and Z8 (Neuromuscular Sprint), which are defined in `zone_physiology.md`. This means the model has no training data for these zones and may hallucinate or refuse when asked about them.

**F-KB2: Knowledge uses "amber" readiness but dataset uses "yellow" and "orange".**
The validation script (`validate_josi.py:37`) uses "amber" readiness, while the dataset generator uses "yellow" and "orange". This naming inconsistency means the model may not respond well to "amber" — a term the user-facing product likely uses.

---

## 8. Summary of Findings

### Critical (should fix before next training run)

| ID | Finding | Impact |
|---|---|---|
| F-TR0 | **Deployed model (SmolLM2) has no training script in repo** | Cannot reproduce or iterate on the production model |
| F-DS2 | Massive duplication via repetition loops | Overfitting to specific phrasings |
| F-TR2 | Manual chat template instead of `apply_chat_template()` | Incorrect formatting may confuse the model |
| F-TR5 | 3B script drops system message | Core behavioral guardrails lost |
| F-TR3 | Wrong model class for 3B text fine-tuning | May load unnecessary components or fail |
| F-DS3 | I6 refusal examples underrepresented | Safety-critical behavior is underweighted |

### Important (should address)

| ID | Finding | Impact |
|---|---|---|
| F-DS1 | Template-driven responses lack diversity | Model may parrot templates instead of reasoning |
| F-DS4 | Near-duplicate leakage into validation set | Validation loss is overly optimistic |
| F-TR1 | pad_token = eos_token | Potential generation artifacts |
| F-TR4 | No validation split for 3B training | Cannot detect overfitting |
| F-TR6 | No early stopping | Risk of overfitting with duplicated data |
| F-TR8 | max_seq_length not passed to SFTConfig | Truncation may not work as intended |
| F-VL3 | I6 contract not validated | Core product requirement not tested |
| F-KB1 | Z7 and Z8 missing from training data | Knowledge gap in model responses |

### Recommended Improvements

| ID | Finding | Impact |
|---|---|---|
| F-DS5 | No documentation on which dataset to use when | Confusing for team members |
| F-DS6 | Persona consistency not validated | Style drift undetected |
| F-TR7 | Aggressive LoRA configuration | May overfit for conversational task |
| F-TR9 | No experiment tracking | Cannot compare training runs |
| F-VL1 | Brittle llama-cli output parsing | Validation may break on updates |
| F-VL2 | Warmth scoring punishes science persona | Metric conflicts with intended behavior |
| F-VL4 | No multi-turn evaluation | Critical behavior untested |
| F-VL5 | Small evaluation set (50 prompts) | Low statistical power |
| F-VL6 | No regression tracking across versions | Cannot detect quality regressions |
| F-DP1 | No post-quantization validation | Quantization degradation undetected |
| F-KB2 | Readiness naming inconsistency | User-facing terms not in training data |

---

## 9. Recommendations (Prioritized)

### Priority 0 — Foundational

1. **Add the SmolLM2 fine-tuning script to version control.** The deployed model (`josi-smollm2-merged-v2-q4_k_m.gguf`) was trained with SmolLM2 (HuggingFaceTB), but no corresponding training script exists in the repo. The existing scripts target Mistral-7B and Ministral-3B, which are not what's running in production. Without the SmolLM2 script, the training process is not reproducible. This script should include the exact hyperparameters, dataset, and LoRA configuration used to produce the v2 model. Note: SmolLM2 is a significantly smaller model (~1.7B parameters) than Mistral-7B, which means the LoRA configuration (rank, target modules) and training hyperparameters likely need to be different.

### Priority 1 — Fix before next training run

2. **Deduplicate the training dataset.** Remove the repetition loops (`for _ in range(N)`) from `generate_dataset.py`. Instead, increase diversity by adding more question/response templates or using paraphrasing. If emphasis on certain behaviors is needed, use sample weights rather than data duplication.

3. **Use `tokenizer.apply_chat_template()` instead of manual formatting.** Replace the `format_chat_message()` function with the tokenizer's built-in template method. This ensures correct formatting and makes the code forward-compatible with other models (including SmolLM2).

4. **Fix the Ministral 3B script.** Use `AutoModelForCausalLM` instead of `Mistral3ForConditionalGeneration`. Include the system message in formatting. Add a validation split.

5. **Rebalance I6 refusal examples.** Increase the number and diversity of I6 refusal examples to be proportional with other task types. This is a safety-critical behavior.

### Priority 2 — Improve quality

6. **Introduce proper train/validation splitting.** Use stratified splitting by task type to ensure the validation set doesn't contain near-duplicates of training examples.

7. **Add Z7 and Z8 to the training data.** These zones are defined in the knowledge base and users may ask about them.

8. **Align readiness terminology.** Use consistent naming (green/amber/red or green/yellow/orange/red) across all datasets and validation scripts.

9. **Enable experiment tracking.** Uncomment and configure Weights & Biases in `requirements.txt` and set `report_to="wandb"`.

10. **Add early stopping.** Configure `EarlyStoppingCallback` with patience of 2-3 evaluations based on validation loss.

### Priority 3 — Strengthen validation

11. **Add I6 contract tests to validation.** Include prompts that directly attempt to get the model to prescribe or modify training plans.

12. **Add multi-turn evaluation.** Test whether the model maintains persona and compliance across conversation turns.

13. **Make warmth scoring persona-aware.** Different personas should have different warmth thresholds.

14. **Run validation on GGUF output.** After quantization, re-run the validation suite to catch degradation.

15. **Expand the test set.** Increase from 50 to at least 100-200 prompts, with more coverage per category.

---

*End of review.*
