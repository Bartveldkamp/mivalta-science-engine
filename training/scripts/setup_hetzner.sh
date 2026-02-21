#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MiValta Josi v6 — Hetzner Server Setup
#
# Run this on the Hetzner GPU server to:
#   1. Check GPU & VRAM availability
#   2. Create Python venv + install dependencies
#   3. Verify all imports work
#
# Prerequisites:
#   - CUDA GPU with >=16GB VRAM (4B) or >=24GB VRAM (8B)
#   - Python 3.10+
#
# Usage:
#   # Full setup (creates venv + installs deps):
#   bash training/scripts/setup_hetzner.sh
#
#   # Just check GPU:
#   bash training/scripts/setup_hetzner.sh --check-only
#
#   # Reinstall deps into existing venv:
#   bash training/scripts/setup_hetzner.sh --deps-only
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAINING_DIR="$REPO_DIR/training"
VENV_DIR="$TRAINING_DIR/venv"

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

    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,cuda_version \
        --format=csv,noheader,nounits | while IFS=, read -r name total free driver cuda; do
        info "GPU: $name"
        info "  VRAM Total: ${total} MB"
        info "  VRAM Free:  ${free} MB"
        info "  Driver: $driver | CUDA: $cuda"

        if [ "${free%.*}" -lt 14000 ]; then
            warn "  Less than 14GB VRAM free. Training needs 16GB+ (4B) or 24GB+ (8B)."
        else
            info "  VRAM OK for training"
        fi
    done
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
# Print usage summary
# =============================================================================
print_ready() {
    VENV_PYTHON="$VENV_DIR/bin/python"
    echo ""
    echo "============================================================"
    echo "  SETUP COMPLETE"
    echo "============================================================"
    echo ""
    echo "  Venv python: $VENV_PYTHON"
    echo ""
    echo "  To train (use screen so it survives SSH disconnect):"
    echo ""
    echo "    screen -dmS train bash -c 'cd $REPO_DIR && $VENV_PYTHON training/scripts/finetune_qwen3.py train --mode unified --model-size 4b 2>&1 | tee training.log'"
    echo ""
    echo "  Check progress:"
    echo ""
    echo "    tail -3 $REPO_DIR/training.log"
    echo ""
    echo "  After training finishes:"
    echo ""
    echo "    $VENV_PYTHON training/scripts/finetune_qwen3.py merge --lora_path ./models/josi-v6-qwen3-4b-unified-*/lora_weights --model-size 4b"
    echo "    $VENV_PYTHON training/scripts/publish_models_v6.py --model ./models/josi-v6-qwen3-4b-unified-*/final"
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
    --deps-only)
        install_deps
        ;;
    *)
        check_gpu
        setup_venv
        install_deps
        print_ready
        ;;
esac
