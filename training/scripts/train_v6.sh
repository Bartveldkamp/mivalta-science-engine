#!/bin/bash
# =============================================================================
# MiValta Josi v6 — One-Command Training Pipeline
# =============================================================================
# Run this on the Hetzner GPU server.
#
# What it does:
#   1. Checks GPU availability
#   2. Fine-tunes Qwen3 (LoRA) on unified interpreter+coach data
#   3. Merges LoRA weights into base model
#   4. Runs sanity checks
#   5. (Optional) Exports to GGUF + publishes
#
# Prerequisites:
#   - GPU with 16GB+ VRAM (4B) or 24GB+ VRAM (8B)
#   - Venv set up: bash training/scripts/setup_hetzner.sh
#
# Usage:
#   cd ~/mivalta/mivalta-science-engine
#
#   # Train 4B (iPhone) — ~2.5 hours on RTX 4000 SFF Ada
#   bash training/scripts/train_v6.sh --model-size 4b
#
#   # Train 8B (Android) — needs 24GB+ VRAM
#   bash training/scripts/train_v6.sh --model-size 8b
#
#   # Train + export GGUF + publish
#   bash training/scripts/train_v6.sh --model-size 4b --publish
#
#   # Inside screen (survives SSH disconnect):
#   screen -dmS train bash -c 'cd ~/mivalta/mivalta-science-engine && bash training/scripts/train_v6.sh --model-size 4b 2>&1 | tee training.log'
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$TRAINING_DIR")"
VENV_PYTHON="$TRAINING_DIR/venv/bin/python"

# Parse args
MODEL_SIZE="4b"
DO_PUBLISH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-size) MODEL_SIZE="$2"; shift 2 ;;
        --publish) DO_PUBLISH=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: Venv not found at $TRAINING_DIR/venv/"
    echo "Run first: bash training/scripts/setup_hetzner.sh"
    exit 1
fi

echo "================================================================"
echo "  MiValta Josi v6 — Training Pipeline"
echo "  Model: Qwen3-${MODEL_SIZE^^}"
echo "  $(date)"
echo "================================================================"

# Check GPU
echo ""
echo "  Checking GPU..."
"$VENV_PYTHON" -c "
import torch
if not torch.cuda.is_available():
    print('  ERROR: No GPU detected! Training requires CUDA.')
    exit(1)
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'  GPU: {name} ({vram:.1f} GB VRAM)')
print(f'  CUDA: {torch.version.cuda}')
print(f'  PyTorch: {torch.__version__}')
"

# ─── Step 1: Train unified model ─────────────────────────────────────────────
echo ""
echo "━━━ Step 1: Training Qwen3-${MODEL_SIZE^^} unified (interpreter + coach) ━━━"

cd "$REPO_DIR"
"$VENV_PYTHON" training/scripts/finetune_qwen3.py train \
    --mode unified \
    --model-size "$MODEL_SIZE"

# Find the latest output directory
OUTPUT_DIR=$(ls -td "$TRAINING_DIR"/models/josi-v6-qwen3-${MODEL_SIZE}-unified-*/ 2>/dev/null | head -1)
if [ -z "$OUTPUT_DIR" ]; then
    # Also check repo-level models dir
    OUTPUT_DIR=$(ls -td "$REPO_DIR"/models/josi-v6-qwen3-${MODEL_SIZE}-unified-*/ 2>/dev/null | head -1)
fi
if [ -z "$OUTPUT_DIR" ]; then
    echo "  ERROR: Training output not found!"
    exit 1
fi
echo "  Output: $OUTPUT_DIR"

# ─── Step 2: Merge LoRA weights ──────────────────────────────────────────────
echo ""
echo "━━━ Step 2: Merging LoRA weights into base model ━━━"

"$VENV_PYTHON" training/scripts/finetune_qwen3.py merge \
    --lora_path "$OUTPUT_DIR/lora_weights" \
    --model-size "$MODEL_SIZE"

# ─── Step 3: Sanity checks ───────────────────────────────────────────────────
echo ""
echo "━━━ Step 3: Sanity checks ━━━"

echo "  Testing interpreter mode..."
"$VENV_PYTHON" training/scripts/finetune_qwen3.py sanity \
    --model_path "$OUTPUT_DIR/merged" \
    --mode interpreter 2>&1 || true

echo "  Testing coach mode..."
"$VENV_PYTHON" training/scripts/finetune_qwen3.py sanity \
    --model_path "$OUTPUT_DIR/merged" \
    --mode coach 2>&1 || true

# ─── Step 4 (optional): Publish ──────────────────────────────────────────────
if $DO_PUBLISH; then
    echo ""
    echo "━━━ Step 4: Publishing (merge -> GGUF -> upload) ━━━"

    "$VENV_PYTHON" training/scripts/publish_models_v6.py \
        --model "$OUTPUT_DIR/final"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Training complete!"
echo ""
echo "  Model: Qwen3-${MODEL_SIZE^^}"
echo "  Output: $OUTPUT_DIR"
echo "  Merged: $OUTPUT_DIR/merged/"
echo ""
if ! $DO_PUBLISH; then
    echo "  To publish:"
    echo "    $VENV_PYTHON training/scripts/publish_models_v6.py --model $OUTPUT_DIR/final"
fi
echo "================================================================"
