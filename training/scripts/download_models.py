#!/usr/bin/env python3
"""
MiValta Josi v6 — Bundle Download Script

Downloads the Josi v6 bundle from the Hetzner training server.
ONE file: josi-v6-bundle.zip containing model GGUF + knowledge.json.

After download, extracts both files to the output directory:
  - josi-v6-q4_k_m.gguf    (~5.0 GB) — single GGUF model
  - knowledge.json           (~153 KB, 114 coaching context cards)

The knowledge cards ship WITH the model — one download, one package.
On-device, the app injects relevant cards into prompts at inference time.

Usage:
    # Download and extract bundle
    python download_models.py

    # Download to custom directory
    python download_models.py --output-dir /path/to/models

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
import zipfile
from pathlib import Path

# Hetzner training server — models served via nginx
SERVER_URL = "http://144.76.62.249/models"

# Bundle filename
BUNDLE_FILE = "josi-v6-bundle.zip"


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
                        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
                        print(
                            f"\r    [{bar}] {pct:5.1f}% ({size_mb:.0f}/{total_mb:.0f} MB)",
                            end="", flush=True,
                        )

            if total > 0:
                print()  # newline after progress bar

        size_bytes = os.path.getsize(dest)
        print(f"    Done ({size_bytes / (1024**3):.2f} GB)")
        return True

    except urllib.error.HTTPError as e:
        print(f"\n    HTTP Error {e.code}: {e.reason}")
        if os.path.exists(dest):
            os.remove(dest)
        return False
    except urllib.error.URLError as e:
        print(f"\n    Connection error: {e.reason}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def fetch_manifest(base_url: str) -> dict | None:
    """Download and parse the model manifest."""
    manifest_url = f"{base_url}/josi-v6-manifest.json"
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
        print("OK")
        return True
    else:
        print("MISMATCH")
        print(f"      Expected: {expected_sha256}")
        print(f"      Actual:   {actual}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Josi v6 bundle (model + knowledge) from the Hetzner training server"
    )

    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: models/gguf/ in project root)")
    parser.add_argument("--server", type=str, default=SERVER_URL,
                        help=f"Base URL for model downloads (default: {SERVER_URL})")
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
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir.parent.parent / "models" / "gguf"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MiValta Josi v6 — Bundle Download")
    print("  One file: model + knowledge cards")
    print("=" * 60)
    print(f"\n  Server: {base_url}")
    print(f"  Output: {output_dir}")

    # Fetch manifest for bundle URL + checksum
    manifest = fetch_manifest(base_url)
    if manifest:
        print(f"  Version: {manifest.get('version', 'unknown')}")
        print(f"  Published: {manifest.get('published', 'unknown')}")

    bundle_meta = (manifest or {}).get("bundle", {"file": BUNDLE_FILE})
    bundle_file = bundle_meta.get("file", BUNDLE_FILE)
    bundle_url = bundle_meta.get("url", f"{base_url}/{bundle_file}")
    bundle_dest = output_dir / bundle_file

    # Check if already extracted (model GGUF exists with correct checksum)
    model_meta = (manifest or {}).get("model", {})
    model_file = model_meta.get("file", "josi-v6-q4_k_m.gguf")
    model_path = output_dir / model_file
    knowledge_path = output_dir / "knowledge.json"

    if model_path.exists() and knowledge_path.exists() and not args.force:
        model_sha = model_meta.get("sha256")
        if model_sha and not args.no_verify:
            if verify_checksum(str(model_path), model_sha):
                size_gb = model_path.stat().st_size / (1024**3)
                print(f"\n  Already downloaded and verified:")
                print(f"    Model:     {model_path} ({size_gb:.2f} GB)")
                print(f"    Knowledge: {knowledge_path}")
                print(f"\n  Use --force to re-download.")
                return 0
        else:
            size_gb = model_path.stat().st_size / (1024**3)
            print(f"\n  Already exists ({size_gb:.2f} GB), skipping (use --force to re-download)")
            return 0

    # ── Download the bundle ─────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Downloading bundle: {bundle_file}")
    print(f"{'─'*60}")

    success = download_file(bundle_url, str(bundle_dest), desc="Josi v6 bundle")
    if not success:
        print("\n  Failed to download bundle")
        return 1

    # Verify bundle checksum
    bundle_sha = bundle_meta.get("sha256")
    if bundle_sha and not args.no_verify:
        if not verify_checksum(str(bundle_dest), bundle_sha):
            print("  Bundle checksum failed! File may be corrupted.")
            print("  Re-run with --force to re-download.")
            return 1

    # ── Extract ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Extracting bundle")
    print(f"{'─'*60}")

    with zipfile.ZipFile(str(bundle_dest), "r") as zf:
        names = zf.namelist()
        print(f"    Contents: {', '.join(names)}")

        for name in names:
            info = zf.getinfo(name)
            size_mb = info.file_size / (1024 * 1024)
            print(f"    Extracting {name} ({size_mb:.1f} MB)...")
            zf.extract(name, str(output_dir))

    # Remove the zip after extraction
    bundle_dest.unlink()
    print(f"    Removed {bundle_file} (extracted)")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Bundle Complete!")
    print(f"{'='*60}")

    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024**3)
        print(f"\n  Model:     {model_path} ({size_gb:.2f} GB)")

    if knowledge_path.exists():
        size_kb = knowledge_path.stat().st_size / 1024
        try:
            with open(knowledge_path) as f:
                kdata = json.load(f)
            entries = kdata.get("total_entries", len(kdata.get("entries", [])))
            print(f"  Knowledge: {knowledge_path} ({size_kb:.0f} KB, {entries} cards)")
        except Exception:
            print(f"  Knowledge: {knowledge_path} ({size_kb:.0f} KB)")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
