# MiValta Josi — Training & Deployment Guide

## Overview

Josi uses a fine-tuned LLM as the on-device coaching personality layer.
The model is downloaded by users on first app launch and runs locally on their device.

**Architecture:**
- GATC/Viterbi/Phase Allocator = authoritative decision engine (deterministic)
- Josi (LLM) = natural language messenger (non-authoritative)
- Coaching cards = bounded knowledge source

**Model versions:**

| Version | Model | Size (Q4_K_M) | Architecture | Status |
|---------|-------|---------------|-------------|--------|
| **v6 (current)** | Qwen3-4B | ~2.5 GB (single file) | Single-model, dual-mode | **Production** |
| v5 (legacy) | Qwen2.5-1.5B-Instruct | ~935 MB x 2 (~1.87 GB total) | Dual-model (interpreter + explainer) | Archived |
| v4 (legacy) | Gemma 3n E2B-it | ~2.8 GB x 2 (~5.6 GB total) | Dual-model | Archived |
| v3 (legacy) | SmolLM2-1.7B / 360M | ~1.0 GB / ~210 MB | Single-model | Archived |

### Why v6

| | v5 (dual Qwen2.5-1.5B) | v6 (single Qwen3-4B) |
|--|--|--|
| Download | 1.87 GB (2 files) | 2.5 GB (1 file) |
| Model intelligence | 1.5B per task | 4B per task (2.7x) |
| Dutch quality | Basic | Excellent (100+ languages native) |
| JSON accuracy | Good (fine-tuned) | Better (stronger base + fine-tuned) |
| Coaching warmth | Okay | Natural, human-like |
| Management | 2 GGUFs, 2 manifests | 1 GGUF, 1 manifest |
| Fits iPhone 8GB | Yes | Yes |
| Fits Samsung 12GB | Yes | Yes |

---

## Qwen3-4B Pipeline (v6 — Current)

### Step 1: Prepare Training Data

v6 uses a **single-model architecture** — one model trained on both interpreter (JSON) and coach (text) tasks. The model switches mode based on the system prompt.

```bash
cd training/scripts

# Create unified dataset (merges interpreter + coach data)
python prepare_v6_data.py

# Optional: update system prompts to v6 versions in the data
python prepare_v6_data.py --update-prompts
```

Output:
- `train_v6_unified.jsonl` — merged + shuffled (~2262 examples)
- `val_v6_unified.jsonl` — merged + shuffled (~241 examples)

### Step 2: Fine-Tune on Hetzner Server

SSH into the Hetzner server and run:

```bash
ssh hetzner
su - cockpit2
cd ~/mivalta-science-engine/training

# Install dependencies (first time only)
pip install -r requirements.txt

# RECOMMENDED: Unified training (both modes in one run)
python scripts/finetune_qwen3.py train --mode unified

# Or train modes separately:
python scripts/finetune_qwen3.py train --mode interpreter
python scripts/finetune_qwen3.py train --mode coach

# Custom params
python scripts/finetune_qwen3.py train --mode unified --lr 1e-5 --epochs 4

# Without W&B tracking
python scripts/finetune_qwen3.py train --mode unified --no_wandb
```

Training takes ~30-60 min on GPU (requires ~16GB VRAM with LoRA on 4B model).
Output goes to `./models/josi-v6-qwen3-{unified,interpreter,coach}-<timestamp>/`.

### Step 3: Merge LoRA Weights

```bash
python scripts/finetune_qwen3.py merge \
  --lora_path ./models/josi-v6-qwen3-unified-<timestamp>/lora_weights
```

Output: `./models/josi-v6-qwen3-unified-<timestamp>/merged/`

### Step 4: Export to GGUF

Two-step process via llama.cpp (Qwen3 uses standard architecture, natively supported):

```bash
# Step 1: Convert to GGUF F16
python ~/llama.cpp/convert_hf_to_gguf.py ./models/.../merged \
  --outfile ./models/gguf/josi-v6-f16.gguf --outtype f16

# Step 2: Quantize to Q4_K_M
~/llama.cpp/build/bin/llama-quantize \
  ./models/gguf/josi-v6-f16.gguf \
  ./models/gguf/josi-v6-q4_k_m.gguf Q4_K_M

# Clean up F16 intermediate
rm ./models/gguf/*-f16.gguf
```

Expected size: ~2.5 GB (Q4_K_M, single file).

### Step 5: Sanity Check

```bash
# Test interpreter mode (JSON output)
python scripts/finetune_qwen3.py sanity \
  --model_path ./models/josi-v6-qwen3-unified-<timestamp>/merged \
  --mode interpreter

# Test coach mode (coaching text output)
python scripts/finetune_qwen3.py sanity \
  --model_path ./models/josi-v6-qwen3-unified-<timestamp>/merged \
  --mode coach
```

Key metrics:
- **Interpreter**: eval_loss < 0.01, 6/6 sanity checks (including Dutch)
- **Coach**: eval_loss < 1.5, 3/3 sanity checks (including Dutch response)

### Step 6: Publish Model (Merge -> GGUF -> Upload)

v6 produces a **single GGUF file** that handles both modes:

| File | Size | Purpose |
|------|------|---------|
| `josi-v6-q4_k_m.gguf` | ~2.5 GB | Both interpreter (JSON) + coach (text) modes |

**Automated publish (recommended):**

```bash
# Full pipeline: merge LoRA -> GGUF Q4_K_M -> publish via nginx
python scripts/publish_models_v6.py \
  --model models/josi-v6-qwen3-unified-<timestamp>/final

# Publish pre-built GGUF file only
python scripts/publish_models_v6.py \
  --gguf models/gguf/josi-v6-q4_k_m.gguf \
  --publish-only

# Merge + convert only (skip publish)
python scripts/publish_models_v6.py \
  --model models/josi-v6-qwen3-unified-<timestamp>/final \
  --no-publish
```

**Developer download:**

```bash
# Direct download
curl -LO http://<server-ip>/models/josi-v6-q4_k_m.gguf
```

**App download flow:**
1. User installs app (~50 MB, no models bundled)
2. First launch: "Setting up your coach..." progress bar
3. Single model downloads from Hetzner server (~2.5 GB)
4. Cached locally, never re-downloaded unless model version updates
5. All inference runs on-device via llama.cpp — **NO network calls during chat**

---

## Android Integration Notes (Qwen3 v6)

- Qwen3 uses `qwen3` architecture — natively supported by llama.cpp
- Chat format: ChatML (`<|im_start|>role\ncontent<|im_end|>`) — same as Qwen2.5
- **Native system role** — ChatML supports system messages directly
- **Single model, two modes** — same GGUF loaded once, different system prompt per mode
- Interpreter: max 200 tokens, temperature 0.3, top_p 0.9
- Coach: max 200 tokens, temperature 0.5, top_p 0.9
- Context cap: 4096 tokens (up from 2048 in v5)
- Dialogue governor: answer-first, max 1 follow-up question per turn

See `docs/JOSI_INTEGRATION_GUIDE.md` for full Kotlin integration spec.

---

## File Structure

```
training/
  scripts/
    finetune_qwen3.py              # v6: Qwen3-4B LoRA fine-tuning (single-model architecture)
    prepare_v6_data.py             # v6: Merge interpreter + coach data into unified dataset
    publish_models_v6.py           # v6: merge -> GGUF -> upload (single model)
    finetune_qwen25.py             # v5 (legacy): Qwen2.5-1.5B dual-model fine-tuning
    prepare_sequential_data.py     # v5 (legacy): Generate sequential explainer data
    publish_models.py              # v5 (legacy): dual-model publish
    download_models.py             # Developer model download with checksum verify
    finetune_gemma3n.py            # Legacy v4: Gemma 3n E2B fine-tuning
    finetune_smollm2.py            # Legacy v3: SmolLM2 fine-tuning
  prompts/
    josi_v6_interpreter.txt        # v6: Interpreter system prompt (multilingual)
    josi_v6_coach.txt              # v6: Coach system prompt (multilingual)
    interpreter_system.txt         # v5 (legacy): Interpreter prompt
    explainer_system.txt           # v5 (legacy): Explainer prompt
    explainer_sequential_system.txt # v5 (legacy): Sequential explainer prompt
  data/
    train_v6_unified.jsonl         # v6: Unified training set (interpreter + coach merged)
    val_v6_unified.jsonl           # v6: Unified validation set
    train_interpreter.jsonl        # Interpreter training set (~1450 examples)
    val_interpreter.jsonl          # Interpreter validation set (~149 examples)
    train_explainer_sequential.jsonl  # v5 Sequential explainer training set (~812 examples)
    val_explainer_sequential.jsonl    # v5 Sequential explainer validation set (~92 examples)
    gold_combined.jsonl            # Source gold data (678 curated)
  requirements.txt
shared/
  llm_intent_parser.py            # JSON post-processor (v3 LLMIntent production)
  gatc_postprocessor.py            # v4/v5/v6 GATCRequest post-processor
  dialogue_governor.py             # Answer-first, max 1 question enforcement
  schemas/
    gatc_request.schema.json       # Interpreter output contract (v4/v5/v6 — unchanged)
    chat_context.schema.json       # Input contract (with athlete_memory)
```

---

## Qwen2.5-1.5B Pipeline (v5, Legacy)

The v5 dual-model pipeline is preserved for reference:
- `finetune_qwen25.py` — LoRA fine-tuning (interpreter + explainer separately)
- `publish_models.py` — Dual-model GGUF publish
- `prepare_sequential_data.py` — Explainer data with interpreter context

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

This changes only the header version number — model weights are identical.
