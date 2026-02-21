#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MiValta Josi v7 — Hetzner Server Setup
#
# Run this on the Hetzner GPU server to:
#   1. Check GPU & VRAM availability
#   2. Create Python venv + install dependencies
#   3. Download Qwen3 base model from HuggingFace
#   4. Verify all imports work
#
# Prerequisites:
#   - CUDA GPU with >=16GB VRAM (4B) or >=24GB VRAM (8B)
#   - Python 3.10+
#
# Usage:
#   # Full setup (creates venv + installs deps + downloads model):
#   bash training/scripts/setup_hetzner.sh
#
#   # Full setup with 8B model (needs >=24GB VRAM):
#   bash training/scripts/setup_hetzner.sh --model-size 8b
#
#   # Just check GPU:
#   bash training/scripts/setup_hetzner.sh --check-only
#
#   # Reinstall deps into existing venv:
#   bash training/scripts/setup_hetzner.sh --deps-only
#
#   # Just download model:
#   bash training/scripts/setup_hetzner.sh --model-only
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAINING_DIR="$REPO_DIR/training"
VENV_DIR="$TRAINING_DIR/venv"
MODELS_DIR="$TRAINING_DIR/models"

# Default model size: 4B (fits RTX 4000 SFF Ada 19.5 GB VRAM)
MODEL_SIZE="${MODEL_SIZE:-4b}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Find python3 binary
find_python() {
    if command -v python3 &> /dev/null; then
        echo "python3"
    elif command -v python &> /dev/null; then
        echo "python"
    else
        error "No python3 or python found. Install Python 3.10+."
        exit 1
    fi
}

PYTHON=$(find_python)

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

    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
        --format=csv,noheader,nounits | while IFS=, read -r name total free driver; do
        info "GPU: $(echo "$name" | xargs)"
        info "  VRAM Total: $(echo "$total" | xargs) MB"
        info "  VRAM Free:  $(echo "$free" | xargs) MB"
        info "  Driver: $(echo "$driver" | xargs)"

        free_clean="$(echo "$free" | xargs)"
        if [ -z "$free_clean" ] || [ "${free_clean%.*}" -lt 14000 ] 2>/dev/null; then
            warn "  Less than 14GB VRAM free. Training needs 16GB+ (4B) or 24GB+ (8B)."
        else
            info "  VRAM OK for training"
        fi
    done || true
}

# =============================================================================
# Step 2: Create venv + install dependencies
# =============================================================================
setup_venv() {
    echo ""
    echo "============================================================"
    echo "  Step 2: Python Virtual Environment"
    echo "============================================================"
    echo ""

    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python" ]; then
        info "Venv already exists: $VENV_DIR"
        info "Using existing venv. Pass --deps-only to reinstall."
    else
        info "Creating venv at: $VENV_DIR"
        $PYTHON -m venv "$VENV_DIR"
        info "Venv created."
    fi

    # Always show the correct python path
    VENV_PYTHON="$VENV_DIR/bin/python"
    info "Venv python: $VENV_PYTHON"
    info "Version: $($VENV_PYTHON --version)"
}

install_deps() {
    echo ""
    echo "============================================================"
    echo "  Step 3: Install Dependencies"
    echo "============================================================"
    echo ""

    VENV_PYTHON="$VENV_DIR/bin/python"
    VENV_PIP="$VENV_DIR/bin/pip"

    if [ ! -f "$VENV_PYTHON" ]; then
        error "Venv not found at $VENV_DIR. Run setup first (no --deps-only)."
        exit 1
    fi

    info "Upgrading pip..."
    "$VENV_PYTHON" -m pip install --upgrade pip

    info "Installing from requirements.txt..."
    "$VENV_PIP" install -r "$TRAINING_DIR/requirements.txt"

    # Verify critical imports
    echo ""
    info "Verifying imports..."
    "$VENV_PYTHON" -c "
import torch
import transformers
import peft
import trl
import datasets
import accelerate

print(f'  torch:          {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'  transformers:   {transformers.__version__}')
print(f'  peft:           {peft.__version__}')
print(f'  trl:            {trl.__version__}')
print(f'  datasets:       {datasets.__version__}')
print(f'  accelerate:     {accelerate.__version__}')

if torch.cuda.is_available():
    print(f'  GPU:            {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  VRAM:           {vram:.1f} GB')
"

    info "All dependencies OK"
}

# =============================================================================
# Step 4: Download Qwen3 base model from HuggingFace
# =============================================================================
download_model() {
    echo ""
    echo "============================================================"
    echo "  Step 4: Download Qwen3 Base Model"
    echo "============================================================"
    echo ""

    VENV_PYTHON="$VENV_DIR/bin/python"

    if [ ! -f "$VENV_PYTHON" ]; then
        error "Venv not found. Run full setup first."
        exit 1
    fi

    # Determine model based on size
    if [ "$MODEL_SIZE" = "8b" ]; then
        HF_MODEL="Qwen/Qwen3-8B"
        LOCAL_DIR="Qwen3-8B"
    else
        HF_MODEL="Qwen/Qwen3-4B"
        LOCAL_DIR="Qwen3-4B"
    fi

    TARGET_DIR="$MODELS_DIR/$LOCAL_DIR"

    if [ -d "$TARGET_DIR" ] && [ -f "$TARGET_DIR/config.json" ]; then
        info "Model already downloaded: $TARGET_DIR"
        info "  $(ls "$TARGET_DIR"/*.safetensors 2>/dev/null | wc -l) safetensors files found"
        return 0
    fi

    mkdir -p "$MODELS_DIR"

    info "Downloading $HF_MODEL → $TARGET_DIR"
    info "This may take 10-30 minutes depending on network speed..."
    echo ""

    # Use huggingface-cli (installed via huggingface_hub in requirements.txt)
    "$VENV_DIR/bin/huggingface-cli" download "$HF_MODEL" \
        --local-dir "$TARGET_DIR" \
        --local-dir-use-symlinks False

    # Verify download
    if [ -f "$TARGET_DIR/config.json" ]; then
        info "Download complete: $TARGET_DIR"
        info "  $(du -sh "$TARGET_DIR" | cut -f1) total size"
        info "  $(ls "$TARGET_DIR"/*.safetensors 2>/dev/null | wc -l) safetensors files"
    else
        error "Download failed — config.json not found in $TARGET_DIR"
        exit 1
    fi
}

# =============================================================================
# Print usage summary
# =============================================================================
print_ready() {
    VENV_PYTHON="$VENV_DIR/bin/python"

    # Determine which model dir to show
    if [ "$MODEL_SIZE" = "8b" ]; then
        LOCAL_DIR="Qwen3-8B"
        SIZE_TAG="8b"
    else
        LOCAL_DIR="Qwen3-4B"
        SIZE_TAG="4b"
    fi

    echo ""
    echo "============================================================"
    echo "  SETUP COMPLETE — Josi v7"
    echo "============================================================"
    echo ""
    echo "  Venv python:  $VENV_PYTHON"
    echo "  Model:        $MODELS_DIR/$LOCAL_DIR"
    echo "  Model size:   $MODEL_SIZE"
    echo ""
    echo "  ── STEP 1: TRAIN ──────────────────────────────────────────"
    echo "  (use screen so it survives SSH disconnect)"
    echo ""
    echo "    screen -dmS train bash -c 'cd $REPO_DIR && \\"
    echo "      $VENV_PYTHON training/scripts/finetune_qwen3.py train \\"
    echo "        --mode unified --model-size $SIZE_TAG 2>&1 | tee training.log'"
    echo ""
    echo "  Check progress:"
    echo "    tail -3 $REPO_DIR/training.log"
    echo ""
    echo "  ── STEP 2: MERGE ──────────────────────────────────────────"
    echo ""
    echo "    $VENV_PYTHON training/scripts/finetune_qwen3.py merge \\"
    echo "      --lora_path ./models/josi-v6-qwen3-${SIZE_TAG}-unified-*/lora_weights \\"
    echo "      --model-size $SIZE_TAG"
    echo ""
    echo "  ── STEP 3: TEST ───────────────────────────────────────────"
    echo ""
    echo "    $VENV_PYTHON training/scripts/finetune_qwen3.py sanity \\"
    echo "      --model_path ./models/josi-v6-qwen3-${SIZE_TAG}-unified-*/merged \\"
    echo "      --mode interpreter"
    echo ""
    echo "    $VENV_PYTHON training/scripts/finetune_qwen3.py chat \\"
    echo "      --model_path ./models/josi-v6-qwen3-${SIZE_TAG}-unified-*/merged"
    echo ""
    echo "  ── STEP 4: PUBLISH ────────────────────────────────────────"
    echo ""
    echo "    $VENV_PYTHON training/scripts/publish_models_v6.py \\"
    echo "      --model ./models/josi-v6-qwen3-${SIZE_TAG}-unified-*/final"
    echo ""
    echo "============================================================"
}

# =============================================================================
# Main
# =============================================================================

# Parse arguments
ACTION=""
while [ $# -gt 0 ]; do
    case "$1" in
        --check-only)   ACTION="check" ;;
        --deps-only)    ACTION="deps" ;;
        --model-only)   ACTION="model" ;;
        --model-size)   shift; MODEL_SIZE="${1:-4b}" ;;
        *)              ;; # ignore unknown
    esac
    shift
done

info "Model size: $MODEL_SIZE"

case "$ACTION" in
    check)
        check_gpu
        ;;
    deps)
        install_deps
        ;;
    model)
        download_model
        ;;
    *)
        check_gpu
        setup_venv
        install_deps
        download_model
        print_ready
        ;;
esac
