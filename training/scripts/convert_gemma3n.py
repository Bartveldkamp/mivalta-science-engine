#!/usr/bin/env python3
"""
MiValta Gemma 3n E2B — GGUF Conversion & Quantization

Converts the fine-tuned (merged) Gemma 3n E2B model to GGUF Q4_K_M
for on-device deployment via llama.cpp on Android.

Pipeline:
  1. HuggingFace model → GGUF F16 (via llama.cpp convert_hf_to_gguf.py)
  2. GGUF F16 → GGUF Q4_K_M (via llama-quantize)

Target: ~2-3 GB Q4_K_M file, 2GB effective RAM on Android.

Usage:
    # Convert merged model to GGUF Q4_K_M (default)
    python convert_gemma3n.py --model_path ./models/josi-v4-gemma3n-merged

    # Multiple quantization levels
    python convert_gemma3n.py --model_path ./models/josi-v4-gemma3n-merged --quant q4_k_m q5_k_m

    # Keep F16 intermediate
    python convert_gemma3n.py --model_path ./models/josi-v4-gemma3n-merged --keep_fp16

    # List quantization options
    python convert_gemma3n.py --list_quant

Requirements:
    - llama.cpp (git clone https://github.com/ggerganov/llama.cpp)
    - Built with: cd llama.cpp && cmake -B build && cmake --build build
"""

import argparse
import re
import shutil
import subprocess
import os
from pathlib import Path


# Quantization options with Gemma 3n E2B size estimates
QUANT_LEVELS = {
    "f16":    {"bits": 16, "size_est": "~10 GB",  "quality": "highest",    "mobile": False},
    "q8_0":   {"bits": 8,  "size_est": "~5.3 GB", "quality": "excellent",  "mobile": False},
    "q5_k_m": {"bits": 5,  "size_est": "~3.5 GB", "quality": "very good",  "mobile": True},
    "q4_k_m": {"bits": 4,  "size_est": "~2.8 GB", "quality": "good",       "mobile": True},   # RECOMMENDED
    "q4_0":   {"bits": 4,  "size_est": "~2.6 GB", "quality": "acceptable", "mobile": True},
    "q3_k_m": {"bits": 3,  "size_est": "~2.2 GB", "quality": "lower",      "mobile": True},
    "q2_k":   {"bits": 2,  "size_est": "~1.8 GB", "quality": "lowest",     "mobile": True},
}

DEFAULT_LLAMA_CPP = Path.home() / "llama.cpp"


def find_llama_cpp():
    """Find llama.cpp installation."""
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


def _extract_pretokenizer_hash(stderr: str):
    """Extract BPE pre-tokenizer hash from llama.cpp conversion error output."""
    match = re.search(r'chkhsh:\s+([0-9a-f]{64})', stderr)
    return match.group(1) if match else None


def _patch_llama_pretokenizer(llama_cpp_path: Path, token_hash: str):
    """Patch convert_hf_to_gguf.py to recognize a fine-tuned model's tokenizer hash.

    Adds the hash before the NotImplementedError raise, mapping it to the
    "default" BPE pre-tokenizer. Creates a .bak backup of the original file.
    Returns the backup path on success, None on failure.
    """
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    backup = convert_script.with_suffix(".py.bak")

    shutil.copy2(convert_script, backup)

    lines = convert_script.read_text().splitlines(keepends=True)

    # The raise sits inside `if res is None:`, so we must insert BEFORE that
    # guard block, not inside it. Find the `if res is None:` line that
    # precedes the NotImplementedError raise.
    guard_idx = None
    raise_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "if res is None:":
            guard_idx = i
        if "raise NotImplementedError" in line and "BPE pre-tokenizer was not recognized" in line:
            raise_idx = i
            break

    # Insert before the guard block if found, otherwise before the raise
    insert_idx = guard_idx if guard_idx is not None else raise_idx
    if insert_idx is None:
        backup.unlink(missing_ok=True)
        return None

    # Detect indentation from the target line
    target_line = lines[insert_idx]
    indent = target_line[: len(target_line) - len(target_line.lstrip())]

    patch_lines = [
        f'{indent}if chkhsh == "{token_hash}":\n',
        f'{indent}    # gemma-3n fine-tuned tokenizer (auto-patched by convert_gemma3n.py)\n',
        f'{indent}    res = "default"\n',
    ]

    lines[insert_idx:insert_idx] = patch_lines
    convert_script.write_text("".join(lines))
    return backup


def _restore_llama_backup(backup_path: Path):
    """Restore convert_hf_to_gguf.py from backup."""
    if backup_path and backup_path.exists():
        original = backup_path.with_suffix(".py")
        shutil.move(str(backup_path), str(original))


def convert_to_gguf(model_path: str, output_path: str, llama_cpp_path: str = None):
    """Convert HuggingFace Gemma model to GGUF format (fp16).

    Automatically patches llama.cpp's converter if the model's BPE
    pre-tokenizer hash is not recognized (common for fine-tuned models).
    """

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

    # Restore from any leftover backup from a previous failed run
    stale_backup = convert_script.with_suffix(".py.bak")
    if stale_backup.exists():
        print("Restoring convert_hf_to_gguf.py from leftover backup...")
        shutil.move(str(stale_backup), str(convert_script))

    print(f"Converting {model_path} to GGUF (Gemma architecture)...")
    print(f"Output: {output_path}")

    cmd = [
        "python", str(convert_script),
        model_path,
        "--outfile", output_path,
        "--outtype", "f16",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        size_gb = os.path.getsize(output_path) / (1024**3)
        print(f"Created: {output_path} ({size_gb:.2f} GB)")
        return output_path

    # Check if the failure is the known pre-tokenizer hash issue
    if "BPE pre-tokenizer was not recognized" not in result.stderr:
        print(f"STDOUT: {result.stdout[-500:]}")
        print(f"STDERR: {result.stderr[-500:]}")
        raise RuntimeError("GGUF conversion failed")

    # Extract the unrecognized hash and auto-patch llama.cpp
    token_hash = _extract_pretokenizer_hash(result.stderr)
    if not token_hash:
        print(f"STDERR: {result.stderr[-1000:]}")
        raise RuntimeError(
            "GGUF conversion failed: BPE pre-tokenizer not recognized and "
            "could not extract hash from error output"
        )

    print(f"Pre-tokenizer hash not in llama.cpp database: {token_hash}")
    print(f"Auto-patching convert_hf_to_gguf.py to register fine-tuned tokenizer...")

    backup = _patch_llama_pretokenizer(llama_cpp, token_hash)
    if not backup:
        raise RuntimeError(
            "GGUF conversion failed: could not auto-patch convert_hf_to_gguf.py. "
            "Manually add this hash to get_vocab_base_pre(): " + token_hash
        )

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout[-500:]}")
            print(f"STDERR: {result.stderr[-500:]}")
            raise RuntimeError("GGUF conversion failed after patching pre-tokenizer")
    finally:
        _restore_llama_backup(backup)

    size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"Created: {output_path} ({size_gb:.2f} GB)")
    return output_path


def quantize_gguf(input_path: str, output_path: str, quant_type: str, llama_cpp_path: str = None):
    """Quantize GGUF model to specified precision."""

    if llama_cpp_path is None:
        llama_cpp_path = find_llama_cpp()

    llama_cpp = Path(llama_cpp_path)

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
            f"Quantize binary not found. Build llama.cpp:\n"
            f"  cd {llama_cpp} && cmake -B build && cmake --build build --target llama-quantize"
        )

    if quant_type not in QUANT_LEVELS:
        raise ValueError(f"Unknown quant type: {quant_type}. Options: {list(QUANT_LEVELS.keys())}")

    info = QUANT_LEVELS[quant_type]
    print(f"Quantizing to {quant_type} (expected: {info['size_est']})...")

    cmd = [str(quantize_bin), input_path, output_path, quant_type]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"STDOUT: {result.stdout[-500:]}")
        print(f"STDERR: {result.stderr[-500:]}")
        raise RuntimeError("Quantization failed")

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
    """Run full conversion: HF Gemma → GGUF F16 → Quantized."""

    if quant_types is None:
        quant_types = ["q4_k_m"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(model_path).name

    # Step 1: Convert to GGUF F16
    fp16_path = output_dir / f"{model_name}-f16.gguf"
    print(f"\n{'='*60}")
    print(f"  Step 1: Convert to GGUF F16")
    print(f"{'='*60}\n")
    convert_to_gguf(model_path, str(fp16_path), llama_cpp_path)

    results = {}

    # Step 2: Quantize
    for quant in quant_types:
        print(f"\n{'='*60}")
        print(f"  Step 2: Quantize to {quant}")
        print(f"{'='*60}\n")

        quant_path = output_dir / f"{model_name}-{quant}.gguf"
        try:
            quantize_gguf(str(fp16_path), str(quant_path), quant, llama_cpp_path)
            results[quant] = str(quant_path)
        except Exception as e:
            print(f"Warning: Failed to quantize {quant}: {e}")

    # Optionally remove F16 intermediate
    if not keep_fp16 and fp16_path.exists():
        print(f"Removing F16 intermediate ({os.path.getsize(str(fp16_path)) / (1024**3):.1f} GB)...")
        os.remove(fp16_path)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Gemma 3n E2B GGUF Export Complete")
    print(f"{'='*60}")
    print(f"\n  Model: {MODEL_ID}")
    print(f"  Source: {model_path}")
    print(f"\n  Created models:")
    for quant, path in results.items():
        info = QUANT_LEVELS[quant]
        size_gb = os.path.getsize(path) / (1024**3)
        print(f"    {quant}: {path}")
        print(f"      Size: {size_gb:.2f} GB | Quality: {info['quality']} | Mobile: {'Yes' if info['mobile'] else 'No'}")

    print(f"\n  Next steps:")
    print(f"    1. Evaluate: python evaluate_gemma3n.py --model {list(results.values())[0]}")
    print(f"    2. Upload to Hetzner Object Storage for app download")
    print(f"    3. Update JOSI_INTEGRATION_GUIDE.md with new model URL")

    return results


MODEL_ID = "google/gemma-3n-E2B-it"


def print_quant_options():
    """Print available quantization options."""
    print(f"\nGemma 3n E2B ({MODEL_ID}) — Quantization Options")
    print("-" * 80)
    print(f"{'Type':<10} {'Bits':<6} {'Est. Size':<12} {'Quality':<14} {'Mobile'}")
    print("-" * 80)
    for name, info in QUANT_LEVELS.items():
        mobile = "Yes" if info["mobile"] else "No"
        marker = " <-- RECOMMENDED" if name == "q4_k_m" else ""
        print(f"{name:<10} {info['bits']:<6} {info['size_est']:<12} {info['quality']:<14} {mobile}{marker}")
    print("-" * 80)
    print("\nRecommended for MiValta on-device:")
    print(f"  q4_k_m (~2.8 GB) — best quality/size balance, fits 2GB effective RAM")
    print(f"  q3_k_m (~2.2 GB) — if storage is tight")


def main():
    parser = argparse.ArgumentParser(
        description="Convert fine-tuned Gemma 3n E2B to GGUF for on-device deployment"
    )

    parser.add_argument("--model_path", type=str,
                        help="Path to merged HuggingFace model directory")
    parser.add_argument("--output_dir", type=str, default="./models/gguf",
                        help="Output directory for GGUF files")
    parser.add_argument("--quant", type=str, nargs="+", default=["q4_k_m"],
                        help="Quantization types (e.g., q4_k_m q5_k_m)")
    parser.add_argument("--llama_cpp_path", type=str, default=None,
                        help="Path to llama.cpp directory")
    parser.add_argument("--keep_fp16", action="store_true",
                        help="Keep the F16 intermediate file")
    parser.add_argument("--list_quant", action="store_true",
                        help="List available quantization options")

    args = parser.parse_args()

    if args.list_quant:
        print_quant_options()
        return

    if not args.model_path:
        parser.error("--model_path is required (unless using --list_quant)")

    full_pipeline(
        model_path=args.model_path,
        output_dir=args.output_dir,
        quant_types=args.quant,
        llama_cpp_path=args.llama_cpp_path,
        keep_fp16=args.keep_fp16,
    )


if __name__ == "__main__":
    main()
