#!/bin/bash
# ============================================================
# MiValta Josi v5 — Training Launcher
#
# Trains both interpreter and explainer models sequentially.
# Requires: setup_gpu.sh has been run first.
#
# Usage:
#   cd ~/mivalta-science-engine/training
#   bash scripts/train_v5.sh                    # Train both
#   bash scripts/train_v5.sh interpreter        # Interpreter only
#   bash scripts/train_v5.sh explainer          # Explainer only
#   bash scripts/train_v5.sh smollm2            # Use SmolLM2-1.7B instead
# ============================================================

set -e

# Force Python UTF-8 mode — TRL's model card template contains non-ASCII
# characters and will crash on servers with ASCII-only locale (no LC_ALL set).
export PYTHONUTF8=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
cd "$TRAINING_DIR"

# Activate venv if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Detect GPU VRAM
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)

echo "============================================================"
echo "  Josi v5 Training — Humanized Data"
echo "  GPU: $GPU_NAME ($VRAM_MB MB)"
echo "============================================================"
echo ""

MODE="${1:-both}"

# Decide model based on VRAM or user choice
if [ "$MODE" = "smollm2" ]; then
    USE_MODEL="smollm2"
    MODE="both"
elif [ "${VRAM_MB:-0}" -ge 20000 ]; then
    USE_MODEL="gemma3n"
else
    USE_MODEL="smollm2"
    echo "NOTE: <20GB VRAM detected, using SmolLM2-1.7B instead of Gemma 3n"
fi

echo "Model: $USE_MODEL"
echo ""

# ============================================================
# TRAIN INTERPRETER
# ============================================================
if [ "$MODE" = "both" ] || [ "$MODE" = "interpreter" ]; then
    echo "============================================================"
    echo "  PHASE 1: Training INTERPRETER model"
    echo "  Data: data/train_interpreter.jsonl (1,450 examples)"
    echo "============================================================"
    echo ""

    if [ "$USE_MODEL" = "gemma3n" ]; then
        python scripts/finetune_gemma3n.py train \
            --mode interpreter \
            --train_data data/train_interpreter.jsonl \
            --val_data data/val_interpreter.jsonl \
            --no-wandb
    else
        python scripts/finetune_smollm2.py train \
            --model_size 1.7B \
            --train_data data/train_interpreter.jsonl \
            --val_data data/val_interpreter.jsonl \
            --no-wandb
    fi

    echo ""
    echo "  Interpreter training COMPLETE"
    echo ""
fi

# ============================================================
# TRAIN EXPLAINER
# ============================================================
if [ "$MODE" = "both" ] || [ "$MODE" = "explainer" ]; then
    echo "============================================================"
    echo "  PHASE 2: Training EXPLAINER model (humanized v5 data)"
    echo "  Data: data/train_explainer_v5.jsonl (1,120 examples)"
    echo "============================================================"
    echo ""

    if [ "$USE_MODEL" = "gemma3n" ]; then
        python scripts/finetune_gemma3n.py train \
            --mode explainer \
            --train_data data/train_explainer_v5.jsonl \
            --val_data data/val_explainer_v5.jsonl \
            --no-wandb
    else
        python scripts/finetune_smollm2.py train \
            --model_size 1.7B \
            --train_data data/train_explainer_v5.jsonl \
            --val_data data/val_explainer_v5.jsonl \
            --no-wandb
    fi

    echo ""
    echo "  Explainer training COMPLETE"
    echo ""
fi

# ============================================================
# POST-TRAINING: List models and next steps
# ============================================================
echo "============================================================"
echo "  TRAINING COMPLETE"
echo ""
echo "  Models saved in: $TRAINING_DIR/models/"
ls -td models/josi-v* 2>/dev/null | head -4 || echo "  (no models found)"
echo ""
echo "  Next steps:"
echo "  1. Merge LoRA weights:"
echo "     python scripts/finetune_gemma3n.py merge --lora_path ./models/<FOLDER>/lora_weights"
echo ""
echo "  2. Export to GGUF:"
echo "     python scripts/export_gguf.py --model ./models/<MERGED_FOLDER>"
echo ""
echo "  3. Evaluate:"
echo "     python scripts/evaluate_gemma3n_v4.py --model ./models/<MERGED_FOLDER>"
echo "============================================================"
