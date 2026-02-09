# MiValta Josi — SmolLM2 Training & Deployment Guide

## Overview

Josi uses a fine-tuned SmolLM2 model as the on-device coaching personality layer.
The model is downloaded by users on first app launch and runs locally on their device.

**Architecture:**
- GATC/Viterbi/Phase Allocator = authoritative decision engine (deterministic)
- Josi (SmolLM2) = natural language messenger (non-authoritative)
- Coaching cards = bounded knowledge source

**Model choice:**
- SmolLM2-1.7B q4_k_m (~1.0 GB) — recommended for production
- SmolLM2-360M q4_k_m (~210 MB) — fallback for low-storage devices

---

## Step 1: Prepare Training Data

Training data is already prepared. To regenerate or modify:

```bash
cd training/scripts

# Preview stats
python prepare_training_data.py --stats

# Generate training file (default: 80 word max)
python prepare_training_data.py

# Custom word limit
python prepare_training_data.py --max-words 60 --output ../data/smollm2_train.jsonl
```

Output: `training/data/smollm2_train.jsonl` (~1,149 examples, avg 53 words)

---

## Step 2: Fine-Tune on Hetzner Server

SSH into the Hetzner server and run:

```bash
ssh hetzner
su - cockpit2
cd ~/mivalta-science-engine/training

# Install dependencies (first time only)
pip install -r requirements.txt

# Fine-tune SmolLM2-1.7B (recommended)
python scripts/finetune_smollm2.py train \
  --model 1.7b \
  --train_data data/smollm2_train.jsonl \
  --epochs 3

# Or SmolLM2-360M (lighter alternative)
python scripts/finetune_smollm2.py train \
  --model 360m \
  --train_data data/smollm2_train.jsonl \
  --epochs 3
```

Training takes ~30-60 min on a GPU server. Output goes to `./models/mivalta-josi-smollm2-1.7b-<timestamp>/`.

---

## Step 3: Merge LoRA Weights

```bash
python scripts/finetune_smollm2.py merge \
  --model 1.7b \
  --lora_path ./models/mivalta-josi-smollm2-1.7b-<timestamp>/lora_weights \
  --output_path ./models/mivalta-josi-smollm2-1.7b-merged
```

---

## Step 4: Export to GGUF

```bash
python scripts/export_gguf.py \
  --model_path ./models/mivalta-josi-smollm2-1.7b-merged \
  --quant q4_k_m

# Output: ./models/gguf/mivalta-josi-smollm2-1.7b-merged-q4_k_m.gguf (~1.0 GB)
```

---

## Step 5: Evaluate

```bash
# Test with GGUF model
python scripts/evaluate_smollm2.py \
  --model ./models/gguf/mivalta-josi-smollm2-1.7b-merged-q4_k_m.gguf \
  --verbose

# Or test HuggingFace model directly (before GGUF export)
python scripts/evaluate_smollm2.py \
  --hf-model ./models/mivalta-josi-smollm2-1.7b-merged \
  --verbose

# Compare 1.7B vs 360M
python scripts/evaluate_smollm2.py \
  --model ./models/gguf/smollm2-1.7b-q4_k_m.gguf \
  --compare ./models/gguf/smollm2-360m-q4_k_m.gguf
```

Key metrics:
- Pass rate > 80%
- Avg warmth > 3.5/5
- Avg brevity > 3.5/5
- No forbidden word leaks
- Pushback rate 5/5 on unrealistic goals

---

## Step 6: Share with Android Developer

The GGUF file needs to be accessible for the developer and eventually for app users.

### For development (share with developer):

```bash
# Option A: Developer has SSH access
# They copy directly:
scp cockpit2@136.243.73.100:~/mivalta-science-engine/training/models/gguf/mivalta-josi-smollm2-1.7b-merged-q4_k_m.gguf .

# Option B: Temporary HTTP download
cd ~/mivalta-science-engine/training/models/gguf
nohup python3 -m http.server 8888 > /dev/null 2>&1 &
# Share: http://136.243.73.100:8888/mivalta-josi-smollm2-1.7b-merged-q4_k_m.gguf
# Kill when done: kill $(lsof -t -i:8888)
```

### For production (user download on app install):

The model should be hosted on a CDN and downloaded on first launch.
Options:
- HuggingFace model repository (free, versioned, CDN-backed)
- AWS S3 / CloudFront
- Firebase Storage (integrates with Android/iOS)
- Your own server with proper caching

**App download flow:**
1. User installs app (~50 MB, no model bundled)
2. First launch: "Setting up your coach..." progress bar
3. Model downloads from CDN (~1.0 GB for 1.7B, ~210 MB for 360M)
4. Cached locally, never re-downloaded unless model version updates
5. All inference runs on-device via llama.cpp

---

## Android Integration Notes

- SmolLM2 uses `llama` architecture — widely supported by Android llama.cpp libraries
- Chat format: ChatML (`<|im_start|>role\ncontent<|im_end|>`)
- System prompt is baked into the training data but should also be set at inference time
- Max generation tokens: 120 (prevents verbose responses)
- Temperature: 0.7 (natural but consistent)

**Prompt format for inference:**
```
<|im_start|>system
You are Josi, a friendly and knowledgeable sports coaching assistant for MiValta. You communicate training decisions made by the coaching engine. Rules: Keep responses under 80 words. Be warm and conversational. Use simple language, not textbook explanations. Ask follow-up questions. Never invent training rules — only explain what the engine decided. Never mention algorithms, GATC, Viterbi, ACWR, or internal systems.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
```

---

## File Structure

```
training/
  scripts/
    finetune_smollm2.py       # Fine-tuning (360M and 1.7B)
    evaluate_smollm2.py        # Validation suite
    export_gguf.py             # GGUF conversion
    prepare_training_data.py   # Dataset preparation
  data/
    smollm2_train.jsonl        # Clean training set (1,149 examples)
    gold_combined.jsonl        # Source gold data
    gold_examples/             # Topic-specific gold data
    philosophy_enhanced.jsonl  # Coaching persona data
  archive/mistral/             # Archived Mistral pipeline (reference only)
  requirements.txt
```

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
