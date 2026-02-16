# MiValta Josi — Training & Deployment Guide

## Overview

Josi uses a fine-tuned LLM as the on-device coaching personality layer.
The model is downloaded by users on first app launch and runs locally on their device.

**Architecture:**
- GATC/Viterbi/Phase Allocator = authoritative decision engine (deterministic)
- Josi (LLM) = natural language messenger (non-authoritative)
- Coaching cards = bounded knowledge source

**Model versions:**

| Version | Model | Size (Q4_K_M) | Status |
|---------|-------|---------------|--------|
| **v4 (current)** | Gemma 3n E2B-it | ~2.8 GB | **Production** |
| v3 (legacy) | SmolLM2-1.7B / 360M | ~1.0 GB / ~210 MB | Archived |

---

## Gemma 3n E2B Pipeline (v4)

### Step 1: Prepare Training Data

v4 uses a **dual-model architecture** — interpreter and explainer trained separately.
Training data is already prepared. To regenerate or modify:

```bash
cd training/scripts

# Convert v3 data to v4 dual-mode format (interpreter + explainer)
python convert_training_data_v4.py --stats          # preview only
python convert_training_data_v4.py                    # generate files

# Augment with edge cases (personas, rare intents)
python augment_training_data_v4.py

# Add memory-enriched examples (~85 examples with athlete_memory context)
python augment_memory_training_data.py --dry-run     # preview
python augment_memory_training_data.py                # append to training data
```

Output files in `training/data/`:
- `train_interpreter.jsonl` — GATCRequest JSON training examples
- `train_explainer.jsonl` — Plain coaching text training examples
- `val_interpreter.jsonl` / `val_explainer.jsonl` — Validation splits

### Step 2: Fine-Tune on Hetzner Server

SSH into the Hetzner server and run:

```bash
ssh hetzner
su - cockpit2
cd ~/mivalta-science-engine/training

# Install dependencies (first time only)
pip install -r requirements.txt

# Fine-tune INTERPRETER (GATCRequest JSON output)
python scripts/finetune_gemma3n.py train --mode interpreter

# Fine-tune EXPLAINER (plain coaching text output)
python scripts/finetune_gemma3n.py train --mode explainer

# Custom params
python scripts/finetune_gemma3n.py train --mode interpreter --lr 3e-5 --epochs 4

# Without W&B tracking
python scripts/finetune_gemma3n.py train --mode explainer --no-wandb
```

Training takes ~1-2 hours per model on a GPU server (requires ~16GB VRAM).
Output goes to `./models/josi-v4-gemma3n-{interpreter,explainer}-<timestamp>/`.

### Step 3: Merge LoRA Weights

```bash
python scripts/finetune_gemma3n.py merge \
  --lora_path ./models/josi-v4-gemma3n-<timestamp>/lora_weights
```

Output: `./models/josi-v4-gemma3n-<timestamp>/merged/`

### Step 4: Export to GGUF

```bash
python scripts/convert_gemma3n.py \
  --model_path ./models/josi-v4-gemma3n-<timestamp>/merged

# Output: ./models/gguf/merged-q4_k_m.gguf (~2.8 GB)
```

### Step 5: Evaluate

v4 evaluates interpreter and explainer separately:

```bash
# Evaluate both models with separate fine-tuned checkpoints
python scripts/evaluate_gemma3n_v4.py \
  --interpreter models/josi-v4-gemma3n-interpreter-<timestamp>/final \
  --explainer models/josi-v4-gemma3n-explainer-<timestamp>/final \
  --verbose

# Evaluate a single mode with GGUF
python scripts/evaluate_gemma3n_v4.py \
  --model ./models/gguf/josi-v4-interpreter-q4_k_m.gguf \
  --mode interpreter --verbose

python scripts/evaluate_gemma3n_v4.py \
  --model ./models/gguf/josi-v4-explainer-q4_k_m.gguf \
  --mode explainer --verbose

# Quick sanity check on merged model (before GGUF export)
python scripts/finetune_gemma3n.py sanity \
  --model_path ./models/josi-v4-gemma3n-interpreter-<timestamp>/merged \
  --mode interpreter
```

Key metrics:
- **Interpreter**: Valid JSON > 95%, action accuracy > 90%, sport/time accuracy > 85%
- **Explainer**: Pass rate > 80%, avg warmth > 3.5/5, avg brevity > 3.5/5
- No forbidden word leaks
- Pushback rate 5/5 on unrealistic goals
- **Governor compliance > 90%** (answer-first, max 1 question)

### Step 6: Publish Models (Merge → GGUF → Upload)

v4 uses a **dual-model architecture** — interpreter + explainer as separate GGUF files:

| Model | File | Purpose |
|-------|------|---------|
| Interpreter | `josi-v4-interpreter-q4_k_m.gguf` | GATCRequest JSON structured output |
| Explainer | `josi-v4-explainer-q4_k_m.gguf` | Plain coaching text output |

**Automated publish (recommended):**

```bash
# Full pipeline: merge LoRA → GGUF Q4_K_M → upload to Hetzner Object Storage
python scripts/publish_models.py \
  --interpreter models/josi-v4-gemma3n-<timestamp>/final \
  --explainer models/josi-v4-gemma3n-<timestamp>/final

# Run in background (persists after terminal close)
nohup python scripts/publish_models.py \
  --interpreter models/josi-v4-gemma3n-<timestamp>/final \
  --explainer models/josi-v4-gemma3n-<timestamp>/final \
  > publish.log 2>&1 &

# Merge + convert only (skip upload)
python scripts/publish_models.py \
  --interpreter models/josi-v4-gemma3n-<timestamp>/final \
  --explainer models/josi-v4-gemma3n-<timestamp>/final \
  --no-upload
```

**Manual upload (if needed):**

```bash
s3cmd put ./models/gguf/josi-v4-interpreter-q4_k_m.gguf \
  s3://mivalta-models/josi-v4-interpreter-q4_k_m.gguf --acl-public
s3cmd put ./models/gguf/josi-v4-explainer-q4_k_m.gguf \
  s3://mivalta-models/josi-v4-explainer-q4_k_m.gguf --acl-public

# Verify
curl -I https://objects.mivalta.com/models/josi-v4-interpreter-q4_k_m.gguf
curl -I https://objects.mivalta.com/models/josi-v4-explainer-q4_k_m.gguf
```

**Developer download:**

```bash
# Automated download with checksum verification
python training/scripts/download_models.py

# Download to custom directory
python training/scripts/download_models.py --output-dir /path/to/models

# Download only one model
python training/scripts/download_models.py --interpreter-only
python training/scripts/download_models.py --explainer-only

# Direct download (no script needed)
curl -LO https://objects.mivalta.com/models/josi-v4-interpreter-q4_k_m.gguf
curl -LO https://objects.mivalta.com/models/josi-v4-explainer-q4_k_m.gguf
```

**App download flow:**
1. User installs app (~50 MB, no models bundled)
2. First launch: "Setting up your coach..." progress bar
3. Both models download from Hetzner Object Storage (~5.6 GB total)
4. Cached locally, never re-downloaded unless model version updates
5. All inference runs on-device via llama.cpp — **NO network calls during chat**

---

## Android Integration Notes (Gemma v4)

- Gemma uses `gemma2` architecture — supported by llama.cpp since mid-2024
- Chat format: Gemma (`<start_of_turn>user\n...<end_of_turn>`)
- **No native system role** — system prompt prepended to first user message
- Max generation tokens: 150
- Temperature: 0.45 (range: 0.4-0.5)
- Context cap: 1024 tokens
- Dialogue governor: answer-first, max 1 follow-up question per turn

See `docs/JOSI_INTEGRATION_GUIDE.md` for full Kotlin integration spec.

---

## File Structure

```
training/
  scripts/
    finetune_gemma3n.py            # QLoRA fine-tuning (Gemma 3n E2B, dual-mode)
    evaluate_gemma3n_v4.py         # Validation suite (interpreter + explainer, memory eval)
    convert_gemma3n.py             # GGUF conversion & quantization
    convert_training_data_v4.py    # v3 → v4 data conversion (LLMIntent → dual-mode)
    augment_training_data_v4.py    # Dataset augmentation (edge cases, diversity)
    augment_memory_training_data.py # Memory-enriched training examples
    publish_models.py              # End-to-end: merge → GGUF → upload to Hetzner
    download_models.py             # Developer model download with checksum verify
    finetune_smollm2.py            # Legacy: SmolLM2 fine-tuning
    evaluate_smollm2.py            # Legacy: SmolLM2 validation
    export_gguf.py                 # Legacy: GGUF conversion
  prompts/
    interpreter_system.txt         # System prompt for interpreter model
    explainer_system.txt           # System prompt for explainer model
  data/
    train_interpreter.jsonl        # v4 Interpreter training set (~1450 examples)
    train_explainer.jsonl          # v4 Explainer training set (~1120 examples)
    val_interpreter.jsonl          # v4 Interpreter validation set
    val_explainer.jsonl            # v4 Explainer validation set
    train_v3.jsonl                 # Legacy v3 source data (8,574 LLMIntent examples)
    val_v3.jsonl                   # Legacy v3 validation data
    gold_combined.jsonl            # Source gold data (678 curated)
    philosophy_enhanced.jsonl      # Coaching persona data
  requirements.txt
shared/
  llm_intent_parser.py            # JSON post-processor (v3 LLMIntent production)
  gatc_postprocessor.py            # v4 GATCRequest post-processor
  memory_extractor.py              # Rule-based fact extraction (v1)
  dialogue_governor.py             # Answer-first, max 1 question enforcement
  schemas/
    llm_intent.schema.json         # Legacy v3 output contract
    gatc_request.schema.json       # v4 Interpreter output contract
    chat_context.schema.json       # v4 Input contract (with athlete_memory)
```

---

## SmolLM2 Pipeline (v3, Legacy)

The SmolLM2 pipeline is preserved for reference and fallback. See the legacy scripts:
- `finetune_smollm2.py` — LoRA fine-tuning
- `evaluate_smollm2.py` — Validation suite
- `export_gguf.py` — GGUF conversion

---

## GGUF Version Note

If the Android llama.cpp library requires GGUF v2, patch the version byte:

```python
import shutil, sys
shutil.copy2(sys.argv[1], sys.argv[2])
with open(sys.argv[2], 'r+b') as f:
    f.seek(4)
    f.write(b'\x02\x00\x00\x00')
```

```bash
python3 -c "..." input.gguf output-v2.gguf
```

This changes only the header version number — model weights are identical.
