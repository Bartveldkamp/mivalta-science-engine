#!/usr/bin/env python3
"""
MiValta Josi v5 — Model Publish Script

End-to-end pipeline: LoRA adapters → merge → GGUF Q4_K_M → publish via nginx.

Produces two GGUF models for the sequential v5 architecture (Qwen2.5-1.5B):
  - josi-v5-interpreter-q4_k_m.gguf  (~935 MB, GATCRequest JSON output)
  - josi-v5-explainer-q4_k_m.gguf    (~935 MB, plain coaching text output)

Publish target: nginx on the Hetzner training server
  URL: http://<server-ip>/models/

Usage:
    # Full pipeline: merge + GGUF + publish
    python publish_models.py \
      --interpreter models/josi-v5-qwen25-interpreter-<timestamp>/final \
      --explainer models/josi-v5-qwen25-explainer-<timestamp>/final

    # Merge + GGUF only (no publish)
    python publish_models.py \
      --interpreter models/josi-v5-qwen25-interpreter-<timestamp>/final \
      --explainer models/josi-v5-qwen25-explainer-<timestamp>/final \
      --no-publish

    # Publish already-converted GGUF files
    python publish_models.py \
      --gguf-interpreter models/gguf/josi-v5-interpreter-q4_k_m.gguf \
      --gguf-explainer models/gguf/josi-v5-explainer-q4_k_m.gguf \
      --publish-only

Requirements:
    - finetune_qwen25.py (merge command)
    - llama.cpp (convert_hf_to_gguf.py + llama-quantize for GGUF conversion)
    - nginx serving /var/www/mivalta-models/ (see training/server/setup_nginx.sh)
    - GPU with ~8GB VRAM (for merge step)
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
GGUF_DIR = PROJECT_ROOT / "models" / "gguf"
MANIFEST_PATH = GGUF_DIR / "manifest.json"

# nginx serve directory — created by training/server/setup_nginx.sh
PUBLISH_DIR = Path("/var/www/mivalta-models")


def get_server_ip() -> str:
    """Get this server's public IP for download URLs."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def get_base_url() -> str:
    """Get the base URL for model downloads."""
    ip = get_server_ip()
    return f"http://{ip}/models"


def sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str], desc: str) -> subprocess.CompletedProcess:
    """Run a command, printing output in real-time."""
    print(f"\n  → {desc}")
    print(f"    $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result


def find_llama_cpp() -> Path | None:
    """Find llama.cpp installation."""
    candidates = [
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path.cwd() / "llama.cpp",
        Path.cwd().parent / "llama.cpp",
    ]
    for loc in candidates:
        if (loc / "convert_hf_to_gguf.py").exists():
            return loc
    return None


def merge_lora(lora_path: str, label: str) -> str:
    """Merge LoRA adapter into base Qwen2.5 model. Returns path to merged model."""
    lora_path = str(Path(lora_path).resolve())
    merged_path = str(Path(lora_path).parent / "merged")

    if Path(merged_path).exists() and (Path(merged_path) / "config.json").exists():
        print(f"\n  ✓ {label} merged model already exists: {merged_path}")
        return merged_path

    run_cmd(
        [sys.executable, str(SCRIPT_DIR / "finetune_qwen25.py"),
         "merge", "--lora_path", lora_path, "--output_path", merged_path],
        f"Merging {label} LoRA into Qwen2.5 base model",
    )
    return merged_path


def convert_to_gguf(merged_path: str, output_name: str, quant: str = "q4_k_m") -> str:
    """Convert merged Qwen2.5 model to GGUF via llama.cpp (2-step: f16 → quantize)."""
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GGUF_DIR / f"{output_name}-{quant}.gguf"

    if output_path.exists():
        size_gb = output_path.stat().st_size / (1024**3)
        print(f"\n  ✓ GGUF already exists: {output_path} ({size_gb:.2f} GB)")
        return str(output_path)

    llama_cpp = find_llama_cpp()
    if llama_cpp is None:
        raise RuntimeError(
            "Could not find llama.cpp. Please clone it:\n"
            "  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp\n"
            "  cd ~/llama.cpp && cmake -B build && cmake --build build"
        )

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    fp16_path = GGUF_DIR / f"{output_name}-f16.gguf"

    # Step 1: Convert HF model to GGUF F16
    run_cmd(
        [sys.executable, str(convert_script), merged_path,
         "--outfile", str(fp16_path), "--outtype", "f16"],
        f"Converting {output_name} to GGUF F16",
    )

    if not fp16_path.exists():
        raise RuntimeError(f"F16 conversion failed: {fp16_path} not created")

    # Step 2: Quantize F16 → Q4_K_M (or other quant type)
    quantize_bin = None
    for candidate in [
        llama_cpp / "build" / "bin" / "llama-quantize",
        llama_cpp / "build" / "llama-quantize",
        llama_cpp / "llama-quantize",
    ]:
        if candidate.exists():
            quantize_bin = candidate
            break

    if quantize_bin is None:
        raise RuntimeError(
            f"llama-quantize not found. Build llama.cpp:\n"
            f"  cd {llama_cpp} && cmake -B build && cmake --build build --target llama-quantize"
        )

    quant_upper = quant.upper()
    run_cmd(
        [str(quantize_bin), str(fp16_path), str(output_path), quant_upper],
        f"Quantizing to {quant_upper}",
    )

    if not output_path.exists():
        raise RuntimeError(f"Quantization failed: {output_path} not created")

    # Clean up F16 intermediate
    if fp16_path.exists():
        fp16_size = fp16_path.stat().st_size / (1024**3)
        print(f"  Removing F16 intermediate ({fp16_size:.2f} GB)...")
        fp16_path.unlink()

    size_gb = output_path.stat().st_size / (1024**3)
    print(f"  Created: {output_path} ({size_gb:.2f} GB)")
    return str(output_path)


def publish_file(local_path: str, filename: str) -> str:
    """Copy file to nginx serve directory. Returns public URL."""
    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    dest = PUBLISH_DIR / filename

    print(f"\n  → Publishing {filename}")
    print(f"    Source: {local_path}")
    print(f"    Dest:   {dest}")

    shutil.copy2(local_path, str(dest))

    size_gb = dest.stat().st_size / (1024**3)
    url = f"{get_base_url()}/{filename}"
    print(f"  ✓ Published ({size_gb:.2f} GB): {url}")
    return url


def write_manifest(models: dict, base_url: str):
    """Write manifest.json with model metadata and download URLs."""
    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "v5",
        "base_model": "Qwen2.5-1.5B-Instruct",
        "published": datetime.now().isoformat(),
        "base_url": base_url,
        "models": models,
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Manifest written: {MANIFEST_PATH}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Publish Josi v5 models: merge → GGUF → nginx"
    )

    # LoRA inputs (for full pipeline)
    parser.add_argument("--interpreter", type=str,
                        help="Path to interpreter LoRA adapter (e.g., models/.../final)")
    parser.add_argument("--explainer", type=str,
                        help="Path to explainer LoRA adapter (e.g., models/.../final)")

    # Pre-built GGUF inputs (for publish-only)
    parser.add_argument("--gguf-interpreter", type=str,
                        help="Path to pre-built interpreter GGUF file")
    parser.add_argument("--gguf-explainer", type=str,
                        help="Path to pre-built explainer GGUF file")

    # Options
    parser.add_argument("--quant", type=str, default="q4_k_m",
                        help="Quantization type (default: q4_k_m)")
    parser.add_argument("--no-publish", action="store_true",
                        help="Skip publishing (only merge/convert)")
    parser.add_argument("--publish-only", action="store_true",
                        help="Only publish (skip merge/convert)")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merge step (use existing merged models)")
    parser.add_argument("--publish-dir", type=str, default=str(PUBLISH_DIR),
                        help=f"nginx serve directory (default: {PUBLISH_DIR})")

    args = parser.parse_args()

    publish_dir = Path(args.publish_dir)

    # Override global PUBLISH_DIR if custom path given
    global PUBLISH_DIR
    PUBLISH_DIR = publish_dir

    # Validate inputs
    if args.publish_only:
        if not args.gguf_interpreter or not args.gguf_explainer:
            parser.error("--publish-only requires --gguf-interpreter and --gguf-explainer")
    elif not args.interpreter or not args.explainer:
        parser.error("--interpreter and --explainer are required (or use --publish-only with --gguf-*)")

    base_url = get_base_url()

    print("=" * 60)
    print("  MiValta Josi v5 — Model Publish Pipeline")
    print(f"  Base model: Qwen2.5-1.5B-Instruct")
    print(f"  Server:     {base_url}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    gguf_files = {}

    if args.publish_only:
        gguf_files["interpreter"] = args.gguf_interpreter
        gguf_files["explainer"] = args.gguf_explainer
    else:
        # Step 1: Merge LoRA adapters
        if not args.skip_merge:
            print(f"\n{'='*60}")
            print("  Step 1: Merge LoRA adapters into base model")
            print(f"{'='*60}")

            merged_interpreter = merge_lora(args.interpreter, "interpreter")
            merged_explainer = merge_lora(args.explainer, "explainer")
        else:
            merged_interpreter = str(Path(args.interpreter).parent / "merged")
            merged_explainer = str(Path(args.explainer).parent / "merged")
            print(f"\n  Skipping merge. Using existing:")
            print(f"    Interpreter: {merged_interpreter}")
            print(f"    Explainer:   {merged_explainer}")

        # Step 2: Convert to GGUF
        print(f"\n{'='*60}")
        print(f"  Step 2: Convert to GGUF {args.quant}")
        print(f"{'='*60}")

        gguf_files["interpreter"] = convert_to_gguf(
            merged_interpreter, "josi-v5-interpreter", args.quant)
        gguf_files["explainer"] = convert_to_gguf(
            merged_explainer, "josi-v5-explainer", args.quant)

    # Compute checksums
    print(f"\n{'='*60}")
    print("  Computing checksums")
    print(f"{'='*60}")

    models_meta = {}
    for role, path in gguf_files.items():
        size_bytes = os.path.getsize(path)
        checksum = sha256_file(path)
        print(f"\n  {role}:")
        print(f"    File: {path}")
        print(f"    Size: {size_bytes / (1024**3):.2f} GB")
        print(f"    SHA-256: {checksum}")

        models_meta[role] = {
            "file": os.path.basename(path),
            "size_bytes": size_bytes,
            "sha256": checksum,
            "quant": args.quant,
        }

    # Step 3: Publish to nginx directory
    if not args.no_publish:
        print(f"\n{'='*60}")
        print(f"  Step 3: Publish to {PUBLISH_DIR}")
        print(f"{'='*60}")

        for role, path in gguf_files.items():
            filename = os.path.basename(path)
            url = publish_file(path, filename)
            models_meta[role]["url"] = url
    else:
        print("\n  Skipping publish (--no-publish)")
        for role, path in gguf_files.items():
            filename = os.path.basename(path)
            models_meta[role]["url"] = f"{base_url}/{filename}"

    # Write manifest
    manifest = write_manifest(models_meta, base_url)

    # Publish manifest too
    if not args.no_publish:
        publish_file(str(MANIFEST_PATH), "josi-v5-manifest.json")

    # Summary
    print(f"\n{'='*60}")
    print("  Publish Complete!")
    print(f"{'='*60}")
    print(f"\n  Models available at:")
    for role, meta in models_meta.items():
        print(f"    {role}: {meta['url']}")
    print(f"    manifest: {base_url}/josi-v5-manifest.json")

    print(f"\n  Developer download:")
    print(f"    python training/scripts/download_models.py")
    print(f"\n  Or direct download:")
    for role, meta in models_meta.items():
        print(f"    curl -LO {meta['url']}")

    print()


if __name__ == "__main__":
    main()
