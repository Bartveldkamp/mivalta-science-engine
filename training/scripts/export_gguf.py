#!/usr/bin/env python3
"""
MiValta GGUF Export Script

Converts fine-tuned Mistral model to GGUF format for llama.cpp.
Supports various quantization levels for mobile deployment.

Requirements:
- llama.cpp (with convert scripts)
- Python 3.10+

Usage:
    # First clone llama.cpp and build it
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && make

    # Then run export
    python export_gguf.py --model_path ./models/mivalta-josi-merged --quant q4_k_m
"""

import argparse
import subprocess
import os
from pathlib import Path
import shutil


# Quantization options for mobile
# Size estimates per model family:
#   size_7b  = Mistral 7B / Ministral 8B
#   size_1.7b = SmolLM2-1.7B
#   size_360m = SmolLM2-360M
QUANT_LEVELS = {
    # Best quality (largest)
    "f16": {"bits": 16, "size_7b": "14GB", "size_1.7b": "3.4GB", "size_360m": "724MB", "quality": "highest", "mobile": False},
    "q8_0": {"bits": 8, "size_7b": "7GB", "size_1.7b": "1.7GB", "size_360m": "360MB", "quality": "excellent", "mobile": False},

    # Balanced (recommended for mobile)
    "q5_k_m": {"bits": 5, "size_7b": "4.5GB", "size_1.7b": "1.2GB", "size_360m": "250MB", "quality": "very good", "mobile": True},
    "q4_k_m": {"bits": 4, "size_7b": "4GB", "size_1.7b": "1.0GB", "size_360m": "210MB", "quality": "good", "mobile": True},  # RECOMMENDED

    # Smallest (lower quality)
    "q4_0": {"bits": 4, "size_7b": "3.8GB", "size_1.7b": "950MB", "size_360m": "200MB", "quality": "acceptable", "mobile": True},
    "q3_k_m": {"bits": 3, "size_7b": "3GB", "size_1.7b": "800MB", "size_360m": "170MB", "quality": "lower", "mobile": True},
    "q2_k": {"bits": 2, "size_7b": "2.5GB", "size_1.7b": "650MB", "size_360m": "140MB", "quality": "lowest", "mobile": True},
}

# Default llama.cpp path
DEFAULT_LLAMA_CPP = Path.home() / "llama.cpp"


def find_llama_cpp():
    """Find llama.cpp installation."""

    # Check common locations
    locations = [
        DEFAULT_LLAMA_CPP,
        Path("/opt/llama.cpp"),
        Path.cwd() / "llama.cpp",
        Path.cwd().parent / "llama.cpp",
    ]

    for loc in locations:
        if (loc / "convert_hf_to_gguf.py").exists():
            return loc

    return None


def convert_to_gguf(
    model_path: str,
    output_path: str,
    llama_cpp_path: str = None,
):
    """Convert HuggingFace model to GGUF format (fp16)."""

    if llama_cpp_path is None:
        llama_cpp_path = find_llama_cpp()
        if llama_cpp_path is None:
            raise RuntimeError(
                "Could not find llama.cpp. Please specify --llama_cpp_path or "
                "clone it: git clone https://github.com/ggerganov/llama.cpp"
            )

    llama_cpp = Path(llama_cpp_path)
    convert_script = llama_cpp / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        raise RuntimeError(f"Convert script not found: {convert_script}")

    print(f"Converting {model_path} to GGUF...")
    print(f"Output: {output_path}")

    cmd = [
        "python", str(convert_script),
        model_path,
        "--outfile", output_path,
        "--outtype", "f16",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError("Conversion failed")

    print(f"Created: {output_path}")
    return output_path


def quantize_gguf(
    input_path: str,
    output_path: str,
    quant_type: str,
    llama_cpp_path: str = None,
):
    """Quantize GGUF model to specified precision."""

    if llama_cpp_path is None:
        llama_cpp_path = find_llama_cpp()

    llama_cpp = Path(llama_cpp_path)

    # Search common binary locations (CMake build, legacy Makefile, alternate names)
    candidates = [
        llama_cpp / "build" / "bin" / "llama-quantize",
        llama_cpp / "build" / "llama-quantize",
        llama_cpp / "llama-quantize",
        llama_cpp / "quantize",
    ]
    quantize_bin = None
    for candidate in candidates:
        if candidate.exists():
            quantize_bin = candidate
            break

    if quantize_bin is None:
        raise RuntimeError(
            f"Quantize binary not found. Build llama.cpp with CMake:\n"
            f"  cd {llama_cpp} && cmake -B build && cmake --build build --target llama-quantize"
        )

    if quant_type not in QUANT_LEVELS:
        raise ValueError(f"Unknown quant type: {quant_type}. Options: {list(QUANT_LEVELS.keys())}")

    print(f"Quantizing to {quant_type}...")
    print(f"Expected size: ~{QUANT_LEVELS[quant_type]['size_7b']}")

    cmd = [str(quantize_bin), input_path, output_path, quant_type]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError("Quantization failed")

    # Report file size
    size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"Created: {output_path} ({size_gb:.2f} GB)")

    return output_path


def full_pipeline(
    model_path: str,
    output_dir: str,
    quant_types: list = None,
    llama_cpp_path: str = None,
    keep_fp16: bool = False,
):
    """Run full conversion pipeline: HF -> GGUF F16 -> Quantized."""

    if quant_types is None:
        quant_types = ["q4_k_m"]  # Default: good mobile balance

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(model_path).name

    # Step 1: Convert to GGUF F16
    fp16_path = output_dir / f"{model_name}-f16.gguf"
    convert_to_gguf(model_path, str(fp16_path), llama_cpp_path)

    results = {}

    # Step 2: Quantize to each level
    for quant in quant_types:
        quant_path = output_dir / f"{model_name}-{quant}.gguf"
        try:
            quantize_gguf(str(fp16_path), str(quant_path), quant, llama_cpp_path)
            results[quant] = str(quant_path)
        except Exception as e:
            print(f"Warning: Failed to quantize {quant}: {e}")

    # Optionally remove F16 (it's large)
    if not keep_fp16 and fp16_path.exists():
        print(f"Removing F16 intermediate file...")
        os.remove(fp16_path)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"\nCreated models:")
    for quant, path in results.items():
        info = QUANT_LEVELS[quant]
        size = os.path.getsize(path) / (1024**3)
        print(f"  {quant}: {path} ({size:.2f} GB)")
        print(f"    Quality: {info['quality']}, Mobile: {'Yes' if info['mobile'] else 'No'}")

    return results


def print_quant_options():
    """Print available quantization options."""
    print("\nAvailable quantization options:")
    print("-" * 90)
    print(f"{'Type':<10} {'Bits':<6} {'Mistral 7B':<12} {'SmolLM2 1.7B':<14} {'SmolLM2 360M':<14} {'Quality':<12} {'Mobile'}")
    print("-" * 90)
    for name, info in QUANT_LEVELS.items():
        mobile = "Yes" if info["mobile"] else "No"
        print(f"{name:<10} {info['bits']:<6} {info['size_7b']:<12} {info['size_1.7b']:<14} {info['size_360m']:<14} {info['quality']:<12} {mobile}")
    print("-" * 90)
    print("\nRecommended for MiValta mobile:")
    print("  SmolLM2-1.7B q4_k_m (~1.0 GB) — best quality/size balance for coaching")
    print("  SmolLM2-360M q4_k_m (~210 MB) — ultra-lightweight for intent/templated responses")


def main():
    parser = argparse.ArgumentParser(
        description="Convert fine-tuned model to GGUF for llama.cpp"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/gguf",
        help="Output directory for GGUF files",
    )
    parser.add_argument(
        "--quant",
        type=str,
        nargs="+",
        default=["q4_k_m"],
        help="Quantization types (e.g., q4_k_m q5_k_m)",
    )
    parser.add_argument(
        "--llama_cpp_path",
        type=str,
        default=None,
        help="Path to llama.cpp directory",
    )
    parser.add_argument(
        "--keep_fp16",
        action="store_true",
        help="Keep the F16 intermediate file",
    )
    parser.add_argument(
        "--list_quant",
        action="store_true",
        help="List available quantization options",
    )

    args = parser.parse_args()

    if args.list_quant:
        print_quant_options()
        return

    full_pipeline(
        model_path=args.model_path,
        output_dir=args.output_dir,
        quant_types=args.quant,
        llama_cpp_path=args.llama_cpp_path,
        keep_fp16=args.keep_fp16,
    )


if __name__ == "__main__":
    main()
