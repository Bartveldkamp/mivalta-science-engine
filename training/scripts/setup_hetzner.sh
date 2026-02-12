#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MiValta Josi v4 — Hetzner Server Setup & Model Download
#
# Run this on the Hetzner GPU server to:
#   1. Check GPU & VRAM availability
#   2. Install Python dependencies
#   3. Accept Gemma license & download google/gemma-3n-E2B-it
#   4. Verify the download
#   5. Prepare for training
#
# Prerequisites:
#   - CUDA GPU with >=16GB VRAM
#   - Python 3.10+
#   - HuggingFace account with Gemma license accepted:
#     https://huggingface.co/google/gemma-3n-E2B-it
#     (Click "Accept" on the model page)
#
# Usage:
#   # First time (full setup):
#   bash setup_hetzner.sh
#
#   # Just download model (deps already installed):
#   bash setup_hetzner.sh --download-only
#
#   # Just check GPU:
#   bash setup_hetzner.sh --check-only
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAINING_DIR="$REPO_DIR/training"
MODELS_DIR="$TRAINING_DIR/models"

MODEL_ID="google/gemma-3n-E2B-it"
MODEL_DIR="$MODELS_DIR/gemma-3n-E2B-it"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Step 1: Check GPU
# =============================================================================
check_gpu() {
    echo ""
    echo "============================================================"
    echo "  Step 1: GPU Check"
    echo "============================================================"
    echo ""

    if ! command -v nvidia-smi &> /dev/null; then
        error "nvidia-smi not found. Is CUDA installed?"
        exit 1
    fi

    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,cuda_version \
        --format=csv,noheader,nounits | while IFS=, read -r name total free driver cuda; do
        info "GPU: $name"
        info "  VRAM Total: ${total} MB"
        info "  VRAM Free:  ${free} MB"
        info "  Driver: $driver | CUDA: $cuda"

        if [ "${free%.*}" -lt 14000 ]; then
            warn "  Less than 14GB VRAM free. QLoRA needs ~16GB."
            warn "  Kill other GPU processes or use a larger GPU."
        else
            info "  VRAM OK for QLoRA training"
        fi
    done

    # Check PyTorch CUDA
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        CUDA_DEV=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        info "PyTorch CUDA: $CUDA_DEV"
    else
        warn "PyTorch can't see CUDA. Will be fixed after pip install."
    fi
}

# =============================================================================
# Step 2: Install Dependencies
# =============================================================================
install_deps() {
    echo ""
    echo "============================================================"
    echo "  Step 2: Install Dependencies"
    echo "============================================================"
    echo ""

    cd "$TRAINING_DIR"

    info "Installing from requirements.txt..."
    pip install -r requirements.txt

    # Verify critical imports
    info "Verifying imports..."
    python3 -c "
import torch
import transformers
import peft
import bitsandbytes
import trl
import datasets

print(f'  torch:          {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'  transformers:   {transformers.__version__}')
print(f'  peft:           {peft.__version__}')
print(f'  bitsandbytes:   {bitsandbytes.__version__}')
print(f'  trl:            {trl.__version__}')
print(f'  datasets:       {datasets.__version__}')
"

    info "All dependencies OK"
}

# =============================================================================
# Step 3: HuggingFace Login & Model Download
# =============================================================================
download_model() {
    echo ""
    echo "============================================================"
    echo "  Step 3: Download Gemma 3n E2B-it"
    echo "============================================================"
    echo ""

    # Check HF login
    if ! python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
        warn "Not logged into HuggingFace."
        echo ""
        echo "  You need a HuggingFace token to download Gemma."
        echo ""
        echo "  1. Go to: https://huggingface.co/google/gemma-3n-E2B-it"
        echo "  2. Click 'Accept' on the license agreement"
        echo "  3. Go to: https://huggingface.co/settings/tokens"
        echo "  4. Create a token with 'Read' access"
        echo ""
        info "Running: huggingface-cli login"
        pip install -q huggingface_hub
        huggingface-cli login
    else
        HF_USER=$(python3 -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
        info "Logged in as: $HF_USER"
    fi

    # Check if model already downloaded
    if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
        info "Model already exists at: $MODEL_DIR"
        info "Verifying..."

        MODEL_SIZE=$(du -sh "$MODEL_DIR" | cut -f1)
        info "  Size: $MODEL_SIZE"

        # Quick validation
        python3 -c "
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained('$MODEL_DIR')
print(f'  Processor: OK')
print(f'  Model: OK')
"
        info "Model verified. Skipping download."
        return 0
    fi

    # Download
    mkdir -p "$MODELS_DIR"
    info "Downloading $MODEL_ID..."
    info "This will be ~10 GB (full precision). Be patient."
    echo ""

    python3 -c "
from huggingface_hub import snapshot_download
import os

path = snapshot_download(
    '$MODEL_ID',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False,
)
print(f'Downloaded to: {path}')

# Show size
total = 0
for dirpath, dirnames, filenames in os.walk(path):
    for f in filenames:
        fp = os.path.join(dirpath, f)
        total += os.path.getsize(fp)
print(f'Total size: {total / (1024**3):.1f} GB')
"

    info "Download complete: $MODEL_DIR"
}

# =============================================================================
# Step 4: Verify Model
# =============================================================================
verify_model() {
    echo ""
    echo "============================================================"
    echo "  Step 4: Verify Model"
    echo "============================================================"
    echo ""

    info "Loading processor + quick inference test..."
    info "Using Gemma3nForConditionalGeneration + AutoProcessor (Gemma 3n API)"

    python3 << 'PYEOF'
import torch
from transformers import Gemma3nForConditionalGeneration, AutoProcessor
import os, sys

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_DIR", "")

print(f"  Model: {MODEL_DIR}")

# Load processor (NOT tokenizer — Gemma 3n uses AutoProcessor)
processor = AutoProcessor.from_pretrained(MODEL_DIR)
print(f"  Processor loaded OK")
print(f"  Chat template: {'yes' if processor.chat_template else 'no'}")

# Test chat template format with Gemma 3n message structure
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are Josi, a coaching assistant."}]},
    {"role": "user", "content": [{"type": "text", "text": "Hello, what is Zone 2 training?"}]},
]
formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"  Template format check:")
print(f"    {formatted[:120]}...")

# Load in bf16 to verify model works
# Note: 4-bit QLoRA is incompatible with Gemma 3n AltUp clamp_() on quantized weights
print(f"\n  Loading model in bf16...")
model = Gemma3nForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
)

params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {params / 1e9:.1f}B (6B raw, 2B effective via MatFormer)")
print(f"  Device: {next(model.parameters()).device}")

# Quick generation test using Gemma 3n message format
test_messages = [
    {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
]
inputs = processor.apply_chat_template(
    test_messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)

input_len = inputs["input_ids"].shape[-1]
response = processor.decode(out[0][input_len:], skip_special_tokens=True)
print(f"  Quick gen test: '{response[:50]}'")

print(f"\n  ✓ Model verified and ready for fine-tuning!")

# Memory report
allocated = torch.cuda.memory_allocated() / (1024**3)
reserved = torch.cuda.memory_reserved() / (1024**3)
print(f"  GPU memory: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")

PYEOF

}

# =============================================================================
# Step 5: Check Training Data
# =============================================================================
check_training_data() {
    echo ""
    echo "============================================================"
    echo "  Step 5: Training Data Check"
    echo "============================================================"
    echo ""

    cd "$TRAINING_DIR"

    TRAIN_FILE="$TRAINING_DIR/data/train_v3.jsonl"
    VAL_FILE="$TRAINING_DIR/data/val_v3.jsonl"

    if [ -f "$TRAIN_FILE" ]; then
        TRAIN_COUNT=$(wc -l < "$TRAIN_FILE")
        info "Training data:   $TRAIN_FILE ($TRAIN_COUNT examples)"
    else
        warn "Training data not found: $TRAIN_FILE"
        warn "Run: python scripts/generate_dataset_v3.py"
    fi

    if [ -f "$VAL_FILE" ]; then
        VAL_COUNT=$(wc -l < "$VAL_FILE")
        info "Validation data: $VAL_FILE ($VAL_COUNT examples)"
    else
        warn "Validation data not found: $VAL_FILE"
    fi

    # Count knowledge cards
    CARD_COUNT=$(find "$REPO_DIR/knowledge/gatc" -name "*.md" 2>/dev/null | wc -l)
    info "Knowledge cards:  $CARD_COUNT files in knowledge/gatc/"
}

# =============================================================================
# Step 6: Ready Summary
# =============================================================================
print_ready() {
    echo ""
    echo "============================================================"
    echo "  SETUP COMPLETE — Ready to Train"
    echo "============================================================"
    echo ""
    echo "  To start fine-tuning:"
    echo ""
    echo "    cd $TRAINING_DIR"
    echo "    python scripts/finetune_gemma3n.py train"
    echo ""
    echo "  Or with custom params:"
    echo ""
    echo "    python scripts/finetune_gemma3n.py train --lr 3e-5 --epochs 4"
    echo ""
    echo "  After training:"
    echo ""
    echo "    python scripts/finetune_gemma3n.py merge --lora_path ./models/josi-v4-gemma3n-*/lora_weights"
    echo "    python scripts/convert_gemma3n.py --model_path ./models/josi-v4-gemma3n-*/merged"
    echo "    python scripts/evaluate_gemma3n.py --hf-model ./models/josi-v4-gemma3n-*/merged --verbose"
    echo ""
    echo "============================================================"
}

# =============================================================================
# Main
# =============================================================================

case "${1:-}" in
    --check-only)
        check_gpu
        ;;
    --download-only)
        download_model
        verify_model
        ;;
    *)
        check_gpu
        install_deps
        download_model
        verify_model
        check_training_data
        print_ready
        ;;
esac
