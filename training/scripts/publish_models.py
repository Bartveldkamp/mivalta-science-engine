#!/usr/bin/env python3
"""
MiValta Josi v5 — Model Publish Script

End-to-end pipeline: LoRA adapters → merge → GGUF Q4_K_M → upload to Hetzner Object Storage.

Produces two GGUF models for the sequential v5 architecture (Qwen2.5-1.5B):
  - josi-v5-interpreter-q4_k_m.gguf  (~935 MB, GATCRequest JSON output)
  - josi-v5-explainer-q4_k_m.gguf    (~935 MB, plain coaching text output)

Upload target: Hetzner Object Storage (S3-compatible)
  URL: https://objects.mivalta.com/models/

Usage:
    # Full pipeline: merge + GGUF + upload
    python publish_models.py \
      --interpreter models/josi-v5-qwen25-interpreter-<timestamp>/final \
      --explainer models/josi-v5-qwen25-explainer-<timestamp>/final

    # Merge + GGUF only (no upload)
    python publish_models.py \
      --interpreter models/josi-v5-qwen25-interpreter-<timestamp>/final \
      --explainer models/josi-v5-qwen25-explainer-<timestamp>/final \
      --no-upload

    # Upload already-converted GGUF files
    python publish_models.py \
      --gguf-interpreter models/gguf/josi-v5-interpreter-q4_k_m.gguf \
      --gguf-explainer models/gguf/josi-v5-explainer-q4_k_m.gguf \
      --upload-only

    # Run in background (persists after terminal close)
    nohup python publish_models.py --interpreter ... --explainer ... > publish.log 2>&1 &

Requirements:
    - finetune_qwen25.py (merge command)
    - llama.cpp (convert_hf_to_gguf.py + llama-quantize for GGUF conversion)
    - s3cmd configured for Hetzner Object Storage (for upload)
    - GPU with ~8GB VRAM (for merge step)
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

# Hetzner Object Storage defaults
HETZNER_S3_HOST = "fsn1.your-objectstorage.com"
HETZNER_S3_HOST_BUCKET = "%(bucket)s.fsn1.your-objectstorage.com"


def ensure_s3cfg():
    """Create ~/.s3cfg from env vars or CLI args if it doesn't exist.

    Supports:
        S3_ACCESS_KEY / S3_SECRET_KEY env vars
        --s3-access-key / --s3-secret-key CLI args (parsed before this call)

    This avoids the need for interactive `s3cmd --configure`.
    """
    s3cfg_path = Path.home() / ".s3cfg"
    if s3cfg_path.exists():
        return  # already configured

    access_key = os.environ.get("S3_ACCESS_KEY", "")
    secret_key = os.environ.get("S3_SECRET_KEY", "")
    host = os.environ.get("S3_ENDPOINT", HETZNER_S3_HOST)
    host_bucket = os.environ.get("S3_HOST_BUCKET", HETZNER_S3_HOST_BUCKET)

    if not access_key or not secret_key:
        print("\n  ERROR: s3cmd is not configured and no credentials provided.")
        print()
        print("  Option 1 — Environment variables (easiest):")
        print("    export S3_ACCESS_KEY='your-access-key'")
        print("    export S3_SECRET_KEY='your-secret-key'")
        print("    python scripts/publish_models.py ...")
        print()
        print("  Option 2 — CLI arguments:")
        print("    python scripts/publish_models.py --s3-access-key KEY --s3-secret-key SECRET ...")
        print()
        print("  Option 3 — Interactive config:")
        print("    s3cmd --configure")
        print()
        print("  Where to get credentials:")
        print("    1. Go to https://console.hetzner.cloud")
        print("    2. Select your project")
        print("    3. Click 'Object Storage' in the left sidebar")
        print("    4. Click 'Manage credentials'")
        print("    5. Generate or copy your Access Key and Secret Key")
        raise SystemExit(1)

    config = f"""[default]
access_key = {access_key}
secret_key = {secret_key}
host_base = {host}
host_bucket = {host_bucket}
use_https = True
signature_v2 = False
"""
    s3cfg_path.write_text(config)
    print(f"\n  Auto-configured s3cmd: {s3cfg_path}")
    print(f"    Endpoint: {host}")


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

    quant_upper = quant.upper().replace("_K_", "_K_")  # Q4_K_M
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


def upload_to_s3(local_path: str, s3_key: str) -> str:
    """Upload file to Hetzner Object Storage. Returns public URL."""
    if not shutil.which("s3cmd"):
        raise RuntimeError(
            "s3cmd not found. Install: apt install s3cmd (or brew install s3cmd)"
        )
    ensure_s3cfg()

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
        "version": "v5",
        "base_model": "Qwen2.5-1.5B-Instruct",
        "published": datetime.now().isoformat(),
        "models": models,
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Manifest written: {MANIFEST_PATH}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Publish Josi v5 models: merge → GGUF → Hetzner Object Storage"
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

    # S3 credentials (alternative to s3cmd --configure)
    parser.add_argument("--s3-access-key", type=str,
                        help="Hetzner Object Storage access key (or set S3_ACCESS_KEY env var)")
    parser.add_argument("--s3-secret-key", type=str,
                        help="Hetzner Object Storage secret key (or set S3_SECRET_KEY env var)")
    parser.add_argument("--s3-endpoint", type=str,
                        help="S3 endpoint (default: fsn1.your-objectstorage.com)")

    args = parser.parse_args()

    # Copy CLI credential args into env vars so ensure_s3cfg() picks them up
    if args.s3_access_key:
        os.environ["S3_ACCESS_KEY"] = args.s3_access_key
    if args.s3_secret_key:
        os.environ["S3_SECRET_KEY"] = args.s3_secret_key
    if args.s3_endpoint:
        os.environ["S3_ENDPOINT"] = args.s3_endpoint

    # Validate inputs
    if args.upload_only:
        if not args.gguf_interpreter or not args.gguf_explainer:
            parser.error("--upload-only requires --gguf-interpreter and --gguf-explainer")
    elif not args.interpreter or not args.explainer:
        parser.error("--interpreter and --explainer are required (or use --upload-only with --gguf-*)")

    print("=" * 60)
    print("  MiValta Josi v5 — Model Publish Pipeline")
    print(f"  Base model: Qwen2.5-1.5B-Instruct")
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
        upload_to_s3(str(MANIFEST_PATH), "josi-v5-manifest.json")

    # Summary
    print(f"\n{'='*60}")
    print("  Publish Complete!")
    print(f"{'='*60}")
    print(f"\n  Models available at:")
    for role, meta in models_meta.items():
        print(f"    {role}: {meta['url']}")
    print(f"    manifest: {S3_PUBLIC_URL}/josi-v5-manifest.json")

    print(f"\n  Developer download:")
    print(f"    python training/scripts/download_models.py")
    print(f"\n  Or direct download:")
    for role, meta in models_meta.items():
        print(f"    curl -LO {meta['url']}")

    print()


if __name__ == "__main__":
    main()
