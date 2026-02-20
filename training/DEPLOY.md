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
| **v5 (current)** | Qwen2.5-1.5B-Instruct | ~935 MB per model (~1.87 GB total) | **Production** |
| v4 (legacy) | Gemma 3n E2B-it | ~2.8 GB per model (~5.6 GB total) | Archived |
| v3 (legacy) | SmolLM2-1.7B / 360M | ~1.0 GB / ~210 MB | Archived |

---

## Qwen2.5-1.5B Pipeline (v5 — Current)

### Step 1: Prepare Training Data

v5 uses a **sequential dual-model architecture** — interpreter and explainer trained separately.
The explainer receives the interpreter's JSON output as context (sequential pipeline).

Training data is already prepared:
- `train_interpreter.jsonl` / `val_interpreter.jsonl` — GATCRequest JSON examples
- `train_explainer_sequential.jsonl` / `val_explainer_sequential.jsonl` — Sequential explainer examples

To regenerate explainer sequential data:

```bash
cd training/scripts
python prepare_sequential_data.py
```

### Step 2: Fine-Tune on Hetzner Server

SSH into the Hetzner server and run:

```bash
ssh hetzner
su - cockpit2
cd ~/mivalta-science-engine/training

# Install dependencies (first time only)
pip install -r requirements.txt

# Fine-tune INTERPRETER (GATCRequest JSON output)
python scripts/finetune_qwen25.py train --mode interpreter

# Fine-tune EXPLAINER (sequential coaching text output)
python scripts/finetune_qwen25.py train --mode explainer

# Custom params
python scripts/finetune_qwen25.py train --mode interpreter --lr 3e-5 --epochs 4

# Without W&B tracking
python scripts/finetune_qwen25.py train --mode explainer --no_wandb
```

Training takes ~20-40 min per model on GPU (requires ~8GB VRAM with LoRA).
Output goes to `./models/josi-v5-qwen25-{interpreter,explainer}-<timestamp>/`.

### Step 3: Merge LoRA Weights

```bash
python scripts/finetune_qwen25.py merge \
  --lora_path ./models/josi-v5-qwen25-<timestamp>/lora_weights
```

Output: `./models/josi-v5-qwen25-<timestamp>/merged/`

### Step 4: Export to GGUF

Two-step process via llama.cpp (Qwen2.5 uses standard architecture, no patches needed):

```bash
# Step 1: Convert to GGUF F16
python ~/llama.cpp/convert_hf_to_gguf.py ./models/.../merged \
  --outfile ./models/gguf/josi-v5-interpreter-f16.gguf --outtype f16

# Step 2: Quantize to Q4_K_M
~/llama.cpp/build/bin/llama-quantize \
  ./models/gguf/josi-v5-interpreter-f16.gguf \
  ./models/gguf/josi-v5-interpreter-q4_k_m.gguf Q4_K_M

# Clean up F16 intermediate
rm ./models/gguf/*-f16.gguf
```

Expected size: ~935 MB per model (Q4_K_M).

### Step 5: Sanity Check

```bash
# Quick sanity check on merged model
python scripts/finetune_qwen25.py sanity \
  --model_path ./models/josi-v5-qwen25-interpreter-<timestamp>/merged \
  --mode interpreter

python scripts/finetune_qwen25.py sanity \
  --model_path ./models/josi-v5-qwen25-explainer-<timestamp>/merged \
  --mode explainer
```

Key metrics:
- **Interpreter**: eval_loss < 0.01, token_accuracy > 99%, 5/5 sanity checks
- **Explainer**: eval_loss < 1.5, token_accuracy > 70%, 2/2 sanity checks

### Step 6: Publish Models (Merge → GGUF → Upload)

v5 uses a **sequential dual-model architecture** — interpreter + explainer as separate GGUF files:

| Model | File | Size | Purpose |
|-------|------|------|---------|
| Interpreter | `josi-v5-interpreter-q4_k_m.gguf` | ~935 MB | GATCRequest JSON structured output |
| Explainer | `josi-v5-explainer-q4_k_m.gguf` | ~935 MB | Plain coaching text output |

**Automated publish (recommended):**

```bash
# Full pipeline: merge LoRA → GGUF Q4_K_M → upload to Hetzner Object Storage
python scripts/publish_models.py \
  --interpreter models/josi-v5-qwen25-interpreter-<timestamp>/final \
  --explainer models/josi-v5-qwen25-explainer-<timestamp>/final

# Upload pre-built GGUF files only
python scripts/publish_models.py \
  --gguf-interpreter models/gguf/josi-v5-interpreter-q4_k_m.gguf \
  --gguf-explainer models/gguf/josi-v5-explainer-q4_k_m.gguf \
  --upload-only

# Merge + convert only (skip upload)
python scripts/publish_models.py \
  --interpreter models/josi-v5-qwen25-interpreter-<timestamp>/final \
  --explainer models/josi-v5-qwen25-explainer-<timestamp>/final \
  --no-upload
```

**S3 credentials (first time):**

```bash
# Option 1: Environment variables
export S3_ACCESS_KEY='your-access-key'
export S3_SECRET_KEY='your-secret-key'

# Option 2: CLI arguments
python scripts/publish_models.py --s3-access-key KEY --s3-secret-key SECRET ...

# Option 3: Configure s3cmd directly
s3cmd --configure

# Get credentials: Hetzner Console → Object Storage → Manage credentials
```

**Manual upload (if needed):**

```bash
s3cmd put ./models/gguf/josi-v5-interpreter-q4_k_m.gguf \
  s3://mivalta-models/josi-v5-interpreter-q4_k_m.gguf --acl-public
s3cmd put ./models/gguf/josi-v5-explainer-q4_k_m.gguf \
  s3://mivalta-models/josi-v5-explainer-q4_k_m.gguf --acl-public

# Verify
curl -I https://objects.mivalta.com/models/josi-v5-interpreter-q4_k_m.gguf
curl -I https://objects.mivalta.com/models/josi-v5-explainer-q4_k_m.gguf
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
curl -LO https://objects.mivalta.com/models/josi-v5-interpreter-q4_k_m.gguf
curl -LO https://objects.mivalta.com/models/josi-v5-explainer-q4_k_m.gguf
```

**App download flow:**
1. User installs app (~50 MB, no models bundled)
2. First launch: "Setting up your coach..." progress bar
3. Both models download from Hetzner Object Storage (~1.87 GB total)
4. Cached locally, never re-downloaded unless model version updates
5. All inference runs on-device via llama.cpp — **NO network calls during chat**

---

## Android Integration Notes (Qwen2.5 v5)

- Qwen2.5 uses `qwen2` architecture — natively supported by llama.cpp
- Chat format: ChatML (`<|im_start|>role\ncontent<|im_end|>`)
- **Native system role** — ChatML supports system messages directly
- Max generation tokens: 150
- Temperature: 0.45 (range: 0.4-0.5)
- Context cap: 2048 tokens
- Dialogue governor: answer-first, max 1 follow-up question per turn

See `docs/JOSI_INTEGRATION_GUIDE.md` for full Kotlin integration spec.

---

## File Structure

```
training/
  scripts/
    finetune_qwen25.py             # v5: Qwen2.5-1.5B LoRA fine-tuning (sequential architecture)
    prepare_sequential_data.py     # v5: Generate sequential explainer training data
    publish_models.py              # v5: merge → GGUF → upload to Hetzner Object Storage
    download_models.py             # v5: Developer model download with checksum verify
    finetune_gemma3n.py            # Legacy v4: Gemma 3n E2B fine-tuning
    evaluate_gemma3n_v4.py         # Legacy v4: Validation suite
    convert_gemma3n.py             # Legacy v4: GGUF conversion
    finetune_smollm2.py            # Legacy v3: SmolLM2 fine-tuning
    evaluate_smollm2.py            # Legacy v3: SmolLM2 validation
    export_gguf.py                 # Generic GGUF conversion via llama.cpp
  prompts/
    interpreter_system.txt         # System prompt for interpreter model
    explainer_system.txt           # System prompt for explainer model
  data/
    train_interpreter.jsonl        # Interpreter training set (~1450 examples)
    val_interpreter.jsonl          # Interpreter validation set (~149 examples)
    train_explainer_sequential.jsonl  # v5 Sequential explainer training set (~812 examples)
    val_explainer_sequential.jsonl    # v5 Sequential explainer validation set (~92 examples)
    train_explainer.jsonl          # Legacy explainer training set (~1120 examples)
    val_explainer.jsonl            # Legacy explainer validation set
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
    gatc_request.schema.json       # v4/v5 Interpreter output contract
    chat_context.schema.json       # v4/v5 Input contract (with athlete_memory)
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
