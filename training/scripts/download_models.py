#!/usr/bin/env python3
"""
MiValta Josi v5 — Model Download Script

Downloads the published Josi v5 GGUF models from the Hetzner training server.
Intended for developers who need the models for local evaluation or integration.

Downloads:
  - josi-v5-interpreter-q4_k_m.gguf  (~935 MB) — GATCRequest JSON output
  - josi-v5-explainer-q4_k_m.gguf    (~935 MB) — Plain coaching text output
  - josi-v5-manifest.json             — Model metadata and checksums

Base model: Qwen2.5-1.5B-Instruct (sequential dual-model architecture)

Usage:
    # Download both models to default location (models/gguf/)
    python download_models.py

    # Download to custom directory
    python download_models.py --output-dir /path/to/models

    # Download only the interpreter
    python download_models.py --interpreter-only

    # Download only the explainer
    python download_models.py --explainer-only

    # Use a different server URL
    python download_models.py --server http://my-server/models

    # Skip checksum verification
    python download_models.py --no-verify

    # Force re-download even if files exist
    python download_models.py --force

Requirements:
    - Python 3.8+ (no external dependencies)
    - Network access to the Hetzner training server
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

# Hetzner training server — models served via nginx
SERVER_URL = "http://144.76.62.249/models"

# Fallback model definitions (used if manifest download fails)
DEFAULT_MODELS = {
    "interpreter": {
        "file": "josi-v5-interpreter-q4_k_m.gguf",
    },
    "explainer": {
        "file": "josi-v5-explainer-q4_k_m.gguf",
    },
}


def sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: str, desc: str = ""):
    """Download a file with progress display."""
    label = desc or os.path.basename(dest)
    print(f"\n  Downloading {label}...")
    print(f"    URL:  {url}")
    print(f"    Dest: {dest}")

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "MiValta-Download/1.0")

        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB chunks

            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        pct = downloaded / total * 100
                        size_mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        bar_len = 30
                        filled = int(bar_len * downloaded / total)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        print(
                            f"\r    [{bar}] {pct:5.1f}% ({size_mb:.0f}/{total_mb:.0f} MB)",
                            end="", flush=True,
                        )

            if total > 0:
                print()  # newline after progress bar

        size_gb = os.path.getsize(dest) / (1024**3)
        print(f"    ✓ Downloaded ({size_gb:.2f} GB)")
        return True

    except urllib.error.HTTPError as e:
        print(f"\n    ✗ HTTP Error {e.code}: {e.reason}")
        if os.path.exists(dest):
            os.remove(dest)
        return False
    except urllib.error.URLError as e:
        print(f"\n    ✗ Connection error: {e.reason}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def fetch_manifest(base_url: str) -> dict | None:
    """Download and parse the model manifest."""
    manifest_url = f"{base_url}/josi-v5-manifest.json"
    try:
        req = urllib.request.Request(manifest_url)
        req.add_header("User-Agent", "MiValta-Download/1.0")
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"  Warning: Could not fetch manifest ({e}), using defaults")
        return None


def verify_checksum(path: str, expected_sha256: str) -> bool:
    """Verify file SHA-256 checksum."""
    print(f"    Verifying checksum...", end=" ", flush=True)
    actual = sha256_file(path)
    if actual == expected_sha256:
        print(f"✓ OK")
        return True
    else:
        print(f"✗ MISMATCH")
        print(f"      Expected: {expected_sha256}")
        print(f"      Actual:   {actual}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Josi v5 GGUF models from the Hetzner training server"
    )

    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: models/gguf/ in project root)")
    parser.add_argument("--server", type=str, default=SERVER_URL,
                        help=f"Base URL for model downloads (default: {SERVER_URL})")
    parser.add_argument("--interpreter-only", action="store_true",
                        help="Download only the interpreter model")
    parser.add_argument("--explainer-only", action="store_true",
                        help="Download only the explainer model")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip SHA-256 checksum verification")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if files exist")

    args = parser.parse_args()
    base_url = args.server.rstrip("/")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Auto-detect project root
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir.parent.parent / "models" / "gguf"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MiValta Josi v5 — Model Download")
    print("  Base model: Qwen2.5-1.5B-Instruct")
    print("=" * 60)
    print(f"\n  Server: {base_url}")
    print(f"  Output: {output_dir}")

    # Fetch manifest for checksums and URLs
    manifest = fetch_manifest(base_url)
    models = (manifest or {}).get("models", DEFAULT_MODELS)

    if manifest:
        print(f"  Published: {manifest.get('published', 'unknown')}")

    # Determine which models to download
    roles = []
    if args.interpreter_only:
        roles = ["interpreter"]
    elif args.explainer_only:
        roles = ["explainer"]
    else:
        roles = ["interpreter", "explainer"]

    results = {}

    for role in roles:
        meta = models.get(role)
        if not meta:
            print(f"\n  ✗ No {role} model found in manifest")
            continue

        filename = meta.get("file", DEFAULT_MODELS[role]["file"])
        url = meta.get("url", f"{base_url}/{filename}")
        dest = output_dir / filename

        # Check if already downloaded
        if dest.exists() and not args.force:
            size_gb = dest.stat().st_size / (1024**3)
            expected_sha = meta.get("sha256")

            if expected_sha and not args.no_verify:
                if verify_checksum(str(dest), expected_sha):
                    print(f"  ✓ {role} already downloaded and verified ({size_gb:.2f} GB)")
                    results[role] = str(dest)
                    continue
                else:
                    print(f"  Checksum mismatch, re-downloading...")
            else:
                print(f"\n  ✓ {role} already exists ({size_gb:.2f} GB), skipping (use --force to re-download)")
                results[role] = str(dest)
                continue

        # Download
        success = download_file(url, str(dest), desc=f"{role} model")
        if not success:
            print(f"\n  ✗ Failed to download {role} model")
            continue

        # Verify checksum
        expected_sha = meta.get("sha256")
        if expected_sha and not args.no_verify:
            if not verify_checksum(str(dest), expected_sha):
                print(f"  ⚠ Checksum verification failed for {role}!")
                print(f"    File may be corrupted. Re-run with --force to re-download.")
                continue

        results[role] = str(dest)

    # Summary
    print(f"\n{'='*60}")
    if len(results) == len(roles):
        print("  Download Complete!")
    else:
        print("  Download finished with errors")
    print(f"{'='*60}")

    for role, path in results.items():
        size_gb = os.path.getsize(path) / (1024**3)
        print(f"\n  {role}: {path} ({size_gb:.2f} GB)")

    if results:
        print(f"\n  Next steps:")
        print(f"    Run sanity checks:")
        if "interpreter" in results:
            print(f"      python training/scripts/finetune_qwen25.py sanity \\")
            print(f"        --model_path {results['interpreter']} --mode interpreter")
        if "explainer" in results:
            print(f"      python training/scripts/finetune_qwen25.py sanity \\")
            print(f"        --model_path {results['explainer']} --mode explainer")
    print()

    return 0 if len(results) == len(roles) else 1


if __name__ == "__main__":
    sys.exit(main())
