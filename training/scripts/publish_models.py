#!/usr/bin/env python3
"""
MiValta Josi v4 — Model Publish Script

End-to-end pipeline: LoRA adapters → merge → GGUF Q4_K_M → upload to Hetzner Object Storage.

Produces two GGUF models for the dual-mode v4 architecture:
  - josi-v4-interpreter-q4_k_m.gguf  (GATCRequest JSON output)
  - josi-v4-explainer-q4_k_m.gguf    (plain coaching text output)

Upload target: Hetzner Object Storage (S3-compatible)
  URL: https://objects.mivalta.com/models/

Usage:
    # Full pipeline: merge + GGUF + upload
    python publish_models.py \
      --interpreter models/josi-v4-gemma3n-20260215_115614/final \
      --explainer models/josi-v4-gemma3n-20260214_215643/final

    # Merge + GGUF only (no upload)
    python publish_models.py \
      --interpreter models/josi-v4-gemma3n-20260215_115614/final \
      --explainer models/josi-v4-gemma3n-20260214_215643/final \
      --no-upload

    # Upload already-converted GGUF files
    python publish_models.py \
      --gguf-interpreter models/gguf/josi-v4-interpreter-q4_k_m.gguf \
      --gguf-explainer models/gguf/josi-v4-explainer-q4_k_m.gguf \
      --upload-only

    # Run in background (persists after terminal close)
    nohup python publish_models.py --interpreter ... --explainer ... > publish.log 2>&1 &

Requirements:
    - finetune_gemma3n.py (merge command)
    - convert_gemma3n.py (GGUF conversion)
    - s3cmd configured for Hetzner Object Storage (for upload)
    - GPU with ~16GB VRAM (for merge step)
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
GGUF_DIR = PROJECT_ROOT / "models" / "gguf"
MANIFEST_PATH = GGUF_DIR / "manifest.json"

S3_BUCKET = "s3://mivalta-models"
S3_PUBLIC_URL = "https://objects.mivalta.com/models"


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


def merge_lora(lora_path: str, label: str) -> str:
    """Merge LoRA adapter into base model. Returns path to merged model."""
    lora_path = str(Path(lora_path).resolve())
    merged_path = str(Path(lora_path).parent / "merged")

    if Path(merged_path).exists() and (Path(merged_path) / "config.json").exists():
        print(f"\n  ✓ {label} merged model already exists: {merged_path}")
        return merged_path

    run_cmd(
        [sys.executable, str(SCRIPT_DIR / "finetune_gemma3n.py"),
         "merge", "--lora_path", lora_path, "--output_path", merged_path],
        f"Merging {label} LoRA into base model",
    )
    return merged_path


def convert_to_gguf(merged_path: str, output_name: str, quant: str = "q4_k_m") -> str:
    """Convert merged model to GGUF. Returns path to quantized GGUF."""
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GGUF_DIR / f"{output_name}-{quant}.gguf"

    if output_path.exists():
        size_gb = output_path.stat().st_size / (1024**3)
        print(f"\n  ✓ GGUF already exists: {output_path} ({size_gb:.2f} GB)")
        return str(output_path)

    run_cmd(
        [sys.executable, str(SCRIPT_DIR / "convert_gemma3n.py"),
         "--model_path", merged_path,
         "--output_dir", str(GGUF_DIR),
         "--output_name", output_name,
         "--quant", quant],
        f"Converting to GGUF {quant}",
    )

    if not output_path.exists():
        raise RuntimeError(f"GGUF conversion did not produce expected output: {output_path}")

    size_gb = output_path.stat().st_size / (1024**3)
    print(f"  Created: {output_path} ({size_gb:.2f} GB)")
    return str(output_path)


def upload_to_s3(local_path: str, s3_key: str) -> str:
    """Upload file to Hetzner Object Storage. Returns public URL."""
    if not shutil.which("s3cmd"):
        raise RuntimeError(
            "s3cmd not found. Install: apt install s3cmd (or brew install s3cmd)\n"
            "Configure: s3cmd --configure (use Hetzner Object Storage credentials)"
        )

    s3_dest = f"{S3_BUCKET}/{s3_key}"
    public_url = f"{S3_PUBLIC_URL}/{s3_key}"

    run_cmd(
        ["s3cmd", "put", local_path, s3_dest, "--acl-public"],
        f"Uploading to {s3_dest}",
    )

    # Verify upload
    result = subprocess.run(
        ["s3cmd", "info", s3_dest],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Upload verification failed for {s3_dest}")

    print(f"  ✓ Uploaded: {public_url}")
    return public_url


def write_manifest(models: dict):
    """Write manifest.json with model metadata and download URLs."""
    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "v4",
        "published": datetime.now().isoformat(),
        "models": models,
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Manifest written: {MANIFEST_PATH}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Publish Josi v4 models: merge → GGUF → Hetzner Object Storage"
    )

    # LoRA inputs (for full pipeline)
    parser.add_argument("--interpreter", type=str,
                        help="Path to interpreter LoRA adapter (e.g., models/.../final)")
    parser.add_argument("--explainer", type=str,
                        help="Path to explainer LoRA adapter (e.g., models/.../final)")

    # Pre-built GGUF inputs (for upload-only)
    parser.add_argument("--gguf-interpreter", type=str,
                        help="Path to pre-built interpreter GGUF file")
    parser.add_argument("--gguf-explainer", type=str,
                        help="Path to pre-built explainer GGUF file")

    # Options
    parser.add_argument("--quant", type=str, default="q4_k_m",
                        help="Quantization type (default: q4_k_m)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip upload to Hetzner Object Storage")
    parser.add_argument("--upload-only", action="store_true",
                        help="Only upload (skip merge/convert)")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merge step (use existing merged models)")

    args = parser.parse_args()

    # Validate inputs
    if args.upload_only:
        if not args.gguf_interpreter or not args.gguf_explainer:
            parser.error("--upload-only requires --gguf-interpreter and --gguf-explainer")
    elif not args.interpreter or not args.explainer:
        parser.error("--interpreter and --explainer are required (or use --upload-only with --gguf-*)")

    print("=" * 60)
    print("  MiValta Josi v4 — Model Publish Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    gguf_files = {}

    if args.upload_only:
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
            merged_interpreter, "josi-v4-interpreter", args.quant)
        gguf_files["explainer"] = convert_to_gguf(
            merged_explainer, "josi-v4-explainer", args.quant)

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

    # Step 3: Upload
    if not args.no_upload:
        print(f"\n{'='*60}")
        print("  Step 3: Upload to Hetzner Object Storage")
        print(f"{'='*60}")

        for role, path in gguf_files.items():
            filename = os.path.basename(path)
            url = upload_to_s3(path, filename)
            models_meta[role]["url"] = url
    else:
        print("\n  Skipping upload (--no-upload)")
        for role, path in gguf_files.items():
            filename = os.path.basename(path)
            models_meta[role]["url"] = f"{S3_PUBLIC_URL}/{filename}"

    # Write manifest
    manifest = write_manifest(models_meta)

    # Upload manifest too
    if not args.no_upload:
        upload_to_s3(str(MANIFEST_PATH), "josi-v4-manifest.json")

    # Summary
    print(f"\n{'='*60}")
    print("  Publish Complete!")
    print(f"{'='*60}")
    print(f"\n  Models available at:")
    for role, meta in models_meta.items():
        print(f"    {role}: {meta['url']}")
    print(f"    manifest: {S3_PUBLIC_URL}/josi-v4-manifest.json")

    print(f"\n  Developer download:")
    print(f"    python training/scripts/download_models.py")
    print(f"\n  Or direct download:")
    for role, meta in models_meta.items():
        print(f"    curl -LO {meta['url']}")

    print()


if __name__ == "__main__":
    main()
