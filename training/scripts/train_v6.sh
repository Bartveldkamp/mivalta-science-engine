#!/bin/bash
# =============================================================================
# MiValta Josi v6 — One-Command Training Pipeline
# =============================================================================
# Run this on the Hetzner GPU server.
#
# What it does:
#   1. Augments training data with cross-domain intelligence examples
#   2. Trains the interpreter (LoRA on Qwen2.5-1.5B)
#   3. Trains the explainer (LoRA on Qwen2.5-1.5B)
#   4. Merges LoRA weights into base model
#   5. Runs sanity checks
#
# Prerequisites:
#   - GPU with 8+ GB VRAM (RTX 3090/4090, A100, etc.)
#   - Python venv with: pip install -r requirements.txt
#   - Internet access (for HuggingFace model download on first run)
#
# Usage:
#   cd ~/mivalta/mivalta-science-engine/training
#   bash scripts/train_v6.sh
#
# To also export to GGUF (requires llama.cpp):
#   bash scripts/train_v6.sh --gguf
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$TRAINING_DIR/data"

echo "================================================================"
echo "  MiValta Josi v6 — Training Pipeline"
echo "  $(date)"
echo "================================================================"

# Check GPU
echo ""
echo "  Checking GPU..."
python3 -c "
import torch
if not torch.cuda.is_available():
    print('  ERROR: No GPU detected! Training requires CUDA.')
    exit(1)
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
print(f'  GPU: {name} ({vram:.1f} GB VRAM)')
if vram < 6:
    print(f'  WARNING: {vram:.1f} GB may be tight. 8+ GB recommended.')
print(f'  CUDA: {torch.version.cuda}')
print(f'  PyTorch: {torch.__version__}')
"

# ─── Step 1: Augment training data ───────────────────────────────────────────
echo ""
echo "━━━ Step 1: Augmenting training data ━━━"
cd "$TRAINING_DIR"
python3 scripts/augment_cross_domain_training.py

# ─── Step 2: Train interpreter ────────────────────────────────────────────────
echo ""
echo "━━━ Step 2: Training INTERPRETER (LoRA on Qwen2.5-1.5B) ━━━"
echo "  Data: $DATA_DIR/train_interpreter_v6.jsonl"

python3 scripts/finetune_qwen25.py train \
    --mode interpreter \
    --train "$DATA_DIR/train_interpreter_v6.jsonl" \
    --val "$DATA_DIR/val_interpreter_v6.jsonl" \
    --no-wandb

# Find the latest interpreter output
INTERP_DIR=$(ls -td "$TRAINING_DIR"/models/josi-v5-qwen25-interpreter-*/ 2>/dev/null | head -1)
if [ -z "$INTERP_DIR" ]; then
    echo "  ERROR: Interpreter training output not found!"
    exit 1
fi
echo "  Interpreter output: $INTERP_DIR"

# ─── Step 3: Train explainer ─────────────────────────────────────────────────
echo ""
echo "━━━ Step 3: Training EXPLAINER (LoRA on Qwen2.5-1.5B) ━━━"
echo "  Data: $DATA_DIR/train_explainer_v6.jsonl"

python3 scripts/finetune_qwen25.py train \
    --mode explainer \
    --train "$DATA_DIR/train_explainer_v6.jsonl" \
    --val "$DATA_DIR/val_explainer_v6.jsonl" \
    --no-wandb

# Find the latest explainer output
EXPL_DIR=$(ls -td "$TRAINING_DIR"/models/josi-v5-qwen25-explainer-*/ 2>/dev/null | head -1)
if [ -z "$EXPL_DIR" ]; then
    echo "  ERROR: Explainer training output not found!"
    exit 1
fi
echo "  Explainer output: $EXPL_DIR"

# ─── Step 4: Merge LoRA weights ──────────────────────────────────────────────
echo ""
echo "━━━ Step 4: Merging LoRA weights into base model ━━━"

echo "  Merging interpreter..."
python3 scripts/finetune_qwen25.py merge --lora_path "$INTERP_DIR/lora_weights"

echo "  Merging explainer..."
python3 scripts/finetune_qwen25.py merge --lora_path "$EXPL_DIR/lora_weights"

# ─── Step 5: Sanity checks ───────────────────────────────────────────────────
echo ""
echo "━━━ Step 5: Sanity checks ━━━"

echo "  Testing interpreter..."
python3 scripts/finetune_qwen25.py sanity \
    --model_path "$INTERP_DIR/merged" \
    --mode interpreter

echo "  Testing explainer..."
python3 scripts/finetune_qwen25.py sanity \
    --model_path "$EXPL_DIR/merged" \
    --mode explainer

# ─── Step 6 (optional): GGUF export ──────────────────────────────────────────
if [[ "${1:-}" == "--gguf" ]]; then
    echo ""
    echo "━━━ Step 6: GGUF export ━━━"

    LLAMA_CPP="$HOME/llama.cpp"
    if [ ! -d "$LLAMA_CPP" ]; then
        echo "  ERROR: llama.cpp not found at $LLAMA_CPP"
        echo "  Clone it: git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp && cd ~/llama.cpp && make"
        exit 1
    fi

    CONVERT="$LLAMA_CPP/convert_hf_to_gguf.py"
    QUANTIZE="$LLAMA_CPP/build/bin/llama-quantize"

    if [ ! -f "$QUANTIZE" ]; then
        QUANTIZE="$LLAMA_CPP/llama-quantize"
    fi

    echo "  Converting interpreter to GGUF..."
    python3 "$CONVERT" "$INTERP_DIR/merged" \
        --outfile "$INTERP_DIR/josi-v6-interpreter-f16.gguf" \
        --outtype f16

    echo "  Quantizing interpreter to Q4_K_M..."
    "$QUANTIZE" "$INTERP_DIR/josi-v6-interpreter-f16.gguf" \
        "$INTERP_DIR/josi-v6-interpreter-q4_k_m.gguf" q4_k_m

    echo "  Converting explainer to GGUF..."
    python3 "$CONVERT" "$EXPL_DIR/merged" \
        --outfile "$EXPL_DIR/josi-v6-explainer-f16.gguf" \
        --outtype f16

    echo "  Quantizing explainer to Q4_K_M..."
    "$QUANTIZE" "$EXPL_DIR/josi-v6-explainer-f16.gguf" \
        "$EXPL_DIR/josi-v6-explainer-q4_k_m.gguf" q4_k_m

    echo ""
    echo "  GGUF models ready:"
    ls -lh "$INTERP_DIR"/josi-v6-*.gguf 2>/dev/null
    ls -lh "$EXPL_DIR"/josi-v6-*.gguf 2>/dev/null
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Training complete!"
echo ""
echo "  Interpreter: $INTERP_DIR"
echo "  Explainer:   $EXPL_DIR"
echo ""
echo "  Merged models in: */merged/"
echo ""
echo "  Next steps:"
echo "    1. Run --gguf flag if you haven't: bash scripts/train_v6.sh --gguf"
echo "    2. Test with simulate.py:"
echo "       python scripts/simulate.py --interactive \\"
echo "         --interpreter $INTERP_DIR/josi-v6-interpreter-q4_k_m.gguf \\"
echo "         --explainer $EXPL_DIR/josi-v6-explainer-q4_k_m.gguf"
echo "    3. Publish: python scripts/publish_models.py"
echo "================================================================"
