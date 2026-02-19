#!/bin/bash
# ============================================================
# MiValta Josi v5 — GPU Training Setup
# Run this on your Hetzner GPU server after cloning the repo.
#
# Usage:
#   cd ~/mivalta-science-engine/training
#   bash scripts/setup_gpu.sh
# ============================================================

set -e

echo "============================================================"
echo "  MiValta Josi v5 — GPU Training Setup"
echo "============================================================"

# -----------------------------------------------------------
# 1. System check
# -----------------------------------------------------------
echo ""
echo "--- System Check ---"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first:"
    echo "  apt install nvidia-driver-550 nvidia-utils-550"
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "VRAM: ${VRAM_MB} MB"

if [ "$VRAM_MB" -lt 16000 ]; then
    echo "WARNING: < 16GB VRAM. Gemma 3n E2B needs ~15GB with LoRA."
    echo "Consider SmolLM2-1.7B instead (~4GB)."
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Installing Python 3..."
    apt update && apt install -y python3 python3-pip python3-venv
fi
python3 --version

# -----------------------------------------------------------
# 2. Virtual environment
# -----------------------------------------------------------
echo ""
echo "--- Setting up Python virtual environment ---"

VENV_DIR="$(dirname "$(realpath "$0")")/../.venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Created venv at $VENV_DIR"
else
    echo "Venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# -----------------------------------------------------------
# 3. Install dependencies
# -----------------------------------------------------------
echo ""
echo "--- Installing dependencies ---"

# PyTorch with CUDA (auto-detect CUDA version)
CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
echo "Detected CUDA: $CUDA_VERSION"

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ML dependencies
pip install -r "$(dirname "$(realpath "$0")")/../requirements.txt"

# Verify CUDA
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('WARNING: CUDA not available in PyTorch!')
    exit(1)
"

# -----------------------------------------------------------
# 4. Download base model
# -----------------------------------------------------------
echo ""
echo "--- Downloading base model ---"

MODEL_DIR="$(dirname "$(realpath "$0")")/../models"
mkdir -p "$MODEL_DIR"

# Check HuggingFace authentication (required for gated models like Gemma)
if [ -z "$HF_TOKEN" ]; then
    # Check if already logged in via huggingface-cli
    if ! python3 -c "from huggingface_hub import HfFolder; assert HfFolder.get_token()" 2>/dev/null; then
        echo ""
        echo "WARNING: HuggingFace authentication required for gated models (e.g. Gemma)."
        echo "  Option 1: export HF_TOKEN=hf_your_token_here"
        echo "  Option 2: huggingface-cli login"
        echo ""
        echo "Get your token at: https://huggingface.co/settings/tokens"
        echo "Accept the model license at: https://huggingface.co/google/gemma-3n-E2B-it"
        echo ""
        read -rp "Enter your HuggingFace token (or press Enter to skip model download): " HF_TOKEN_INPUT
        if [ -n "$HF_TOKEN_INPUT" ]; then
            export HF_TOKEN="$HF_TOKEN_INPUT"
        fi
    fi
fi

if [ "$VRAM_MB" -ge 20000 ]; then
    MODEL_CHOICE="gemma3n"
    echo "VRAM >= 20GB: Using Gemma 3n E2B (2B effective, best quality)"
    if [ ! -d "$MODEL_DIR/gemma-3n-E2B-it" ]; then
        if [ -z "$HF_TOKEN" ] && ! python3 -c "from huggingface_hub import HfFolder; assert HfFolder.get_token()" 2>/dev/null; then
            echo "Skipping Gemma download (no HF token). Run again after authenticating."
        else
            echo "Downloading google/gemma-3n-E2B-it..."
            python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('google/gemma-3n-E2B-it', local_dir='$MODEL_DIR/gemma-3n-E2B-it')
print('Download complete')
"
        fi
    else
        echo "Model already downloaded"
    fi
elif [ "$VRAM_MB" -ge 8000 ]; then
    MODEL_CHOICE="smollm2-1.7b"
    echo "VRAM 8-20GB: Using SmolLM2-1.7B (lighter, still good)"
    echo "Model will be downloaded automatically by HuggingFace on first run"
else
    MODEL_CHOICE="smollm2-360m"
    echo "VRAM < 8GB: Using SmolLM2-360M (smallest, limited quality)"
    echo "Model will be downloaded automatically by HuggingFace on first run"
fi

# -----------------------------------------------------------
# 5. Verify training data
# -----------------------------------------------------------
echo ""
echo "--- Verifying training data ---"

DATA_DIR="$(dirname "$(realpath "$0")")/../data"

check_file() {
    if [ -f "$DATA_DIR/$1" ]; then
        LINES=$(wc -l < "$DATA_DIR/$1")
        echo "  OK: $1 ($LINES examples)"
    else
        echo "  MISSING: $1"
    fi
}

check_file "train_interpreter.jsonl"
check_file "val_interpreter.jsonl"
check_file "train_explainer_v5.jsonl"
check_file "val_explainer_v5.jsonl"

# -----------------------------------------------------------
# 6. Build llama.cpp for GGUF export
# -----------------------------------------------------------
echo ""
echo "--- Setting up llama.cpp for GGUF export ---"

LLAMA_DIR="$HOME/llama.cpp"
if [ ! -d "$LLAMA_DIR" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
    cd "$LLAMA_DIR"
    cmake -B build -DGGML_CUDA=ON
    cmake --build build --config Release -j$(nproc)
    cd -
    echo "llama.cpp built with CUDA"
else
    echo "llama.cpp already exists at $LLAMA_DIR"
fi

# -----------------------------------------------------------
# Done
# -----------------------------------------------------------
echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Model: $MODEL_CHOICE"
echo "  Venv:  $VENV_DIR"
echo ""
echo "  Next: Run training with:"
echo "    source $VENV_DIR/bin/activate"
echo "    cd $(dirname "$(realpath "$0")")/.."
echo "    bash scripts/train_v5.sh"
echo "============================================================"
