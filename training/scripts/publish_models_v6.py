#!/usr/bin/env python3
"""
MiValta Josi v6 — Single Model Publish Script

End-to-end pipeline: LoRA adapter → merge → GGUF Q4_K_M → publish via nginx.

Produces ONE downloadable file for the v6 single-model architecture:
  - josi-v6-bundle.zip  (~5.0 GB) — contains:
      - josi-v6-q4_k_m.gguf  (GGUF model, stored without compression)
      - knowledge.json  (~153 KB, coaching context cards, deflated)
  - josi-v6-manifest.json  (version, checksums, download URL — fetched separately)

The app downloads ONE zip file. Model + knowledge ship together as one package.
Knowledge is updated atomically with the model — same version, same download, same file.

Publish target: nginx on the Hetzner training server
  URL: http://<server-ip>/models/

Usage:
    # Full pipeline: merge + GGUF + publish (includes knowledge.json)
    python publish_models_v6.py --model models/josi-v6-qwen3-unified-<timestamp>/final

    # Merge + GGUF only (no publish)
    python publish_models_v6.py --model models/josi-v6-qwen3-unified-<timestamp>/final --no-publish

    # Publish already-converted GGUF file
    python publish_models_v6.py --gguf models/gguf/josi-v6-q4_k_m.gguf --publish-only

Requirements:
    - finetune_qwen3.py (merge command)
    - llama.cpp (convert_hf_to_gguf.py + llama-quantize for GGUF conversion)
    - nginx serving /var/www/mivalta-models/ (see training/server/setup_nginx.sh)
    - GPU with ~24GB VRAM (for merge step of 8B model) or ~16GB (4B)
    - knowledge/generated/knowledge.json (run export_knowledge_json.py first)
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
GGUF_DIR = PROJECT_ROOT / "models" / "gguf"
MANIFEST_PATH = GGUF_DIR / "manifest.json"
KNOWLEDGE_JSON = PROJECT_ROOT / "knowledge" / "generated" / "knowledge.json"

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
    print(f"\n  -> {desc}")
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


def merge_lora(lora_path: str) -> str:
    """Merge LoRA adapter into base Qwen3 model. Returns path to merged model."""
    lora_path = str(Path(lora_path).resolve())
    merged_path = str(Path(lora_path).parent / "merged")

    if Path(merged_path).exists() and (Path(merged_path) / "config.json").exists():
        print(f"\n  Merged model already exists: {merged_path}")
        return merged_path

    run_cmd(
        [sys.executable, str(SCRIPT_DIR / "finetune_qwen3.py"),
         "merge", "--lora_path", lora_path, "--output_path", merged_path],
        "Merging LoRA into Qwen3 base model",
    )
    return merged_path


def convert_to_gguf(merged_path: str, output_name: str, quant: str = "q4_k_m") -> str:
    """Convert merged Qwen3 model to GGUF via llama.cpp (2-step: f16 -> quantize)."""
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GGUF_DIR / f"{output_name}-{quant}.gguf"

    if output_path.exists():
        size_gb = output_path.stat().st_size / (1024**3)
        print(f"\n  GGUF already exists: {output_path} ({size_gb:.2f} GB)")
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

    # Step 2: Quantize F16 -> Q4_K_M
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

    print(f"\n  -> Publishing {filename}")
    print(f"    Source: {local_path}")
    print(f"    Dest:   {dest}")

    shutil.copy2(local_path, str(dest))

    size_gb = dest.stat().st_size / (1024**3)
    url = f"{get_base_url()}/{filename}"
    print(f"  Published ({size_gb:.2f} GB): {url}")
    return url


def write_manifest(model_meta: dict, knowledge_meta: dict | None,
                   bundle_meta: dict, base_url: str):
    """Write manifest.json with bundle download URL + individual file metadata."""
    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "v6",
        "architecture": "single-model, dual-mode (Qwen3)",
        "base_model": "Qwen3",
        "published": datetime.now().isoformat(),
        "base_url": base_url,
        "bundle": bundle_meta,
        "model": model_meta,
        "upgrade_note": "v6 replaces v5 dual-model (2x Qwen2.5-1.5B) with single Qwen3",
    }

    if knowledge_meta:
        manifest["knowledge"] = knowledge_meta

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Manifest written: {MANIFEST_PATH}")
    return manifest


def main():
    global PUBLISH_DIR

    parser = argparse.ArgumentParser(
        description="Publish Josi v6 model: merge -> GGUF -> nginx (single model)"
    )

    # LoRA input (for full pipeline)
    parser.add_argument("--model", type=str,
                        help="Path to LoRA adapter (e.g., models/.../final)")

    # Pre-built GGUF input (for publish-only)
    parser.add_argument("--gguf", type=str,
                        help="Path to pre-built GGUF file")

    # Options
    parser.add_argument("--quant", type=str, default="q4_k_m",
                        help="Quantization type (default: q4_k_m)")
    parser.add_argument("--no-publish", action="store_true",
                        help="Skip publishing (only merge/convert)")
    parser.add_argument("--publish-only", action="store_true",
                        help="Only publish (skip merge/convert)")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merge step (use existing merged model)")
    parser.add_argument("--publish-dir", type=str, default=str(PUBLISH_DIR),
                        help=f"nginx serve directory (default: {PUBLISH_DIR})")

    args = parser.parse_args()

    PUBLISH_DIR = Path(args.publish_dir)

    # Validate inputs
    if args.publish_only:
        if not args.gguf:
            parser.error("--publish-only requires --gguf")
    elif not args.model:
        parser.error("--model is required (or use --publish-only with --gguf)")

    base_url = get_base_url()

    print("=" * 60)
    print("  MiValta Josi v6 — Single Model Publish Pipeline")
    print(f"  Base model: Qwen3")
    print(f"  Architecture: single-model, dual-mode")
    print(f"  Server:     {base_url}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    gguf_path = None

    if args.publish_only:
        gguf_path = args.gguf
    else:
        # Step 1: Merge LoRA adapter
        if not args.skip_merge:
            print(f"\n{'='*60}")
            print("  Step 1: Merge LoRA adapter into base model")
            print(f"{'='*60}")
            merged_path = merge_lora(args.model)
        else:
            merged_path = str(Path(args.model).parent / "merged")
            print(f"\n  Skipping merge. Using: {merged_path}")

        # Step 2: Convert to GGUF
        print(f"\n{'='*60}")
        print(f"  Step 2: Convert to GGUF {args.quant}")
        print(f"{'='*60}")
        gguf_path = convert_to_gguf(merged_path, "josi-v6", args.quant)

    # Compute checksum
    print(f"\n{'='*60}")
    print("  Computing checksum")
    print(f"{'='*60}")

    size_bytes = os.path.getsize(gguf_path)
    checksum = sha256_file(gguf_path)
    print(f"\n  Model: {gguf_path}")
    print(f"  Size: {size_bytes / (1024**3):.2f} GB")
    print(f"  SHA-256: {checksum}")

    model_meta = {
        "file": os.path.basename(gguf_path),
        "size_bytes": size_bytes,
        "sha256": checksum,
        "quant": args.quant,
    }

    # Bundle knowledge.json
    knowledge_meta = None
    print(f"\n{'='*60}")
    print("  Bundling knowledge.json")
    print(f"{'='*60}")

    if KNOWLEDGE_JSON.exists():
        knowledge_size = os.path.getsize(str(KNOWLEDGE_JSON))
        knowledge_checksum = sha256_file(str(KNOWLEDGE_JSON))

        # Read entry count from the JSON
        with open(KNOWLEDGE_JSON) as f:
            knowledge_data = json.load(f)
        entry_count = knowledge_data.get("total_entries", 0)

        knowledge_meta = {
            "file": "knowledge.json",
            "size_bytes": knowledge_size,
            "sha256": knowledge_checksum,
            "entries": entry_count,
        }
        print(f"\n  Knowledge: {KNOWLEDGE_JSON}")
        print(f"  Size: {knowledge_size / 1024:.1f} KB ({entry_count} entries)")
        print(f"  SHA-256: {knowledge_checksum}")
    else:
        print(f"\n  WARNING: {KNOWLEDGE_JSON} not found!")
        print(f"  Run: python knowledge/scripts/export_knowledge_json.py")
        print(f"  Continuing without knowledge bundle...")

    # Step 3: Create single zip bundle (model + knowledge in one file)
    print(f"\n{'='*60}")
    print("  Step 3: Create bundle zip (one file download)")
    print(f"{'='*60}")

    bundle_name = "josi-v6-bundle.zip"
    bundle_path = GGUF_DIR / bundle_name

    print(f"\n  -> Creating {bundle_name}")
    print(f"    Model:     {os.path.basename(gguf_path)}")
    if KNOWLEDGE_JSON.exists():
        print(f"    Knowledge: knowledge.json")

    with zipfile.ZipFile(str(bundle_path), "w") as zf:
        # GGUF is already compressed (quantized) — store without compression
        print(f"    Adding model (stored, no compression)...")
        zf.write(gguf_path, os.path.basename(gguf_path), compress_type=zipfile.ZIP_STORED)

        # Knowledge.json is small text — compress it
        if KNOWLEDGE_JSON.exists():
            print(f"    Adding knowledge.json (deflated)...")
            zf.write(str(KNOWLEDGE_JSON), "knowledge.json", compress_type=zipfile.ZIP_DEFLATED)

    bundle_size = os.path.getsize(str(bundle_path))
    bundle_checksum = sha256_file(str(bundle_path))
    print(f"  Bundle: {bundle_path} ({bundle_size / (1024**3):.2f} GB)")
    print(f"  SHA-256: {bundle_checksum}")

    bundle_meta = {
        "file": bundle_name,
        "size_bytes": bundle_size,
        "sha256": bundle_checksum,
    }

    # Step 4: Publish to nginx directory
    if not args.no_publish:
        print(f"\n{'='*60}")
        print(f"  Step 4: Publish to {PUBLISH_DIR}")
        print(f"{'='*60}")

        bundle_url = publish_file(str(bundle_path), bundle_name)
        bundle_meta["url"] = bundle_url
    else:
        print("\n  Skipping publish (--no-publish)")
        bundle_meta["url"] = f"{base_url}/{bundle_name}"

    # Write manifest (bundle-based: one download URL)
    manifest = write_manifest(model_meta, knowledge_meta, bundle_meta, base_url)

    # Publish manifest
    if not args.no_publish:
        publish_file(str(MANIFEST_PATH), "josi-v6-manifest.json")

    # Summary
    print(f"\n{'='*60}")
    print("  Publish Complete!")
    print(f"{'='*60}")
    print(f"\n  Bundle:   {bundle_meta.get('url', bundle_name)} ({bundle_size / (1024**3):.2f} GB)")
    print(f"  Contains: {os.path.basename(gguf_path)} + knowledge.json")
    print(f"  Manifest: {base_url}/josi-v6-manifest.json")

    print(f"\n  App downloads on first launch:")
    print(f"    1. GET {base_url}/josi-v6-manifest.json  (version check, ~1 KB)")
    print(f"    2. GET {bundle_meta.get('url', base_url + '/' + bundle_name)}  (model + knowledge, one file)")
    print(f"    3. Stream-extract zip -> GGUF + knowledge.json on device")

    print()


if __name__ == "__main__":
    main()
