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

Training data is already prepared. To regenerate or modify:

```bash
cd training/scripts

# Preview stats
python prepare_training_data.py --stats

# Generate training file (default: 80 word max)
python prepare_training_data.py

# Custom word limit for Gemma (100 word target)
python prepare_training_data.py --max-words 100 --output ../data/gemma3n_train.jsonl
```

### Step 2: Fine-Tune on Hetzner Server

SSH into the Hetzner server and run:

```bash
ssh hetzner
su - cockpit2
cd ~/mivalta-science-engine/training

# Install dependencies (first time only)
pip install -r requirements.txt

# Fine-tune Gemma 3n E2B with QLoRA
python scripts/finetune_gemma3n.py train

# Custom params
python scripts/finetune_gemma3n.py train --lr 3e-5 --epochs 4

# Without W&B tracking
python scripts/finetune_gemma3n.py train --no-wandb
```

Training takes ~1-2 hours on a GPU server (requires ~16GB VRAM).
Output goes to `./models/josi-v4-gemma3n-<timestamp>/`.

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

```bash
# Test with GGUF model
python scripts/evaluate_gemma3n.py \
  --model ./models/gguf/merged-q4_k_m.gguf \
  --verbose

# Or test HuggingFace model directly (before GGUF export)
python scripts/evaluate_gemma3n.py \
  --hf-model ./models/josi-v4-gemma3n-<timestamp>/merged \
  --verbose

# Large-scale eval (1000 prompts)
python scripts/evaluate_gemma3n.py \
  --hf-model ./models/josi-v4-gemma3n-<timestamp>/merged \
  --prompts-file data/test_prompts_1000.json

# Compare Gemma vs SmolLM2
python scripts/evaluate_gemma3n.py \
  --hf-model ./models/josi-v4-gemma3n-<timestamp>/merged \
  --compare-hf ./models/josi-v3-360M-merged
```

Key metrics:
- Pass rate > 80%
- Avg warmth > 3.5/5
- Avg brevity > 3.5/5
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
    finetune_gemma3n.py        # QLoRA fine-tuning (Gemma 3n E2B)
    evaluate_gemma3n.py        # Validation suite (with dialogue governor checks)
    convert_gemma3n.py         # GGUF conversion & quantization
    publish_models.py          # End-to-end: merge → GGUF → upload to Hetzner
    download_models.py         # Developer model download with checksum verify
    finetune_smollm2.py        # Legacy: SmolLM2 fine-tuning (360M and 1.7B)
    evaluate_smollm2.py        # Legacy: SmolLM2 validation
    export_gguf.py             # Legacy: GGUF conversion
    prepare_training_data.py   # Dataset preparation (shared)
    generate_dataset_v3.py     # Synthetic data generation (shared)
  data/
    train_v3.jsonl             # Training set (8,574 examples)
    val_v3.jsonl               # Validation set (1,440 examples)
    smollm2_train.jsonl        # Legacy SmolLM2 training set
    gold_combined.jsonl        # Source gold data
    gold_examples/             # Topic-specific gold data
    philosophy_enhanced.jsonl  # Coaching persona data
  archive/mistral/             # Archived Mistral pipeline (reference only)
  requirements.txt
shared/
  llm_intent_parser.py        # JSON post-processor (production)
  dialogue_governor.py         # Answer-first, max 1 question enforcement
  schemas/
    llm_intent.schema.json     # Output contract
    chat_context.schema.json   # Input contract
    tool_dispatch.json         # Tool routing & tier allowlists
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
