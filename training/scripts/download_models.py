#!/usr/bin/env python3
"""
MiValta Josi v6 — Model + Knowledge Download Script

Downloads the Josi v6 model bundle from the Hetzner training server.
The bundle always includes model + knowledge as one atomic package.

Downloads:
  - josi-v6-q4_k_m.gguf    (~5.0 GB 8B / ~2.5 GB 4B) — single GGUF model
  - knowledge.json           (~153 KB, 114 coaching context cards)
  - josi-v6-manifest.json    — version, checksums, download URLs

The knowledge cards ship WITH the model — they are not optional.
On-device, the app injects relevant cards into prompts at inference time.

Usage:
    # Download model + knowledge bundle
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
from pathlib import Path

# Hetzner training server — models served via nginx
SERVER_URL = "http://144.76.62.249/models"

# Fallback definitions (used if manifest download fails)
DEFAULT_MODEL = {
    "file": "josi-v6-q4_k_m.gguf",
}
DEFAULT_KNOWLEDGE = {
    "file": "knowledge.json",
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
                        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
                        print(
                            f"\r    [{bar}] {pct:5.1f}% ({size_mb:.0f}/{total_mb:.0f} MB)",
                            end="", flush=True,
                        )

            if total > 0:
                print()  # newline after progress bar

        size_bytes = os.path.getsize(dest)
        if size_bytes > 1024 * 1024:
            print(f"    Done ({size_bytes / (1024**3):.2f} GB)")
        else:
            print(f"    Done ({size_bytes / 1024:.1f} KB)")
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
        print(f"OK")
        return True
    else:
        print(f"MISMATCH")
        print(f"      Expected: {expected_sha256}")
        print(f"      Actual:   {actual}")
        return False


def download_and_verify(meta: dict, base_url: str, output_dir: Path,
                        default_file: str, label: str,
                        force: bool, no_verify: bool) -> str | None:
    """Download a file from the bundle, verify checksum. Returns local path or None."""
    filename = meta.get("file", default_file)
    url = meta.get("url", f"{base_url}/{filename}")
    dest = output_dir / filename

    # Check if already downloaded
    if dest.exists() and not force:
        expected_sha = meta.get("sha256")
        if expected_sha and not no_verify:
            if verify_checksum(str(dest), expected_sha):
                size = dest.stat().st_size
                if size > 1024 * 1024:
                    print(f"  {label} already downloaded and verified ({size / (1024**3):.2f} GB)")
                else:
                    print(f"  {label} already downloaded and verified ({size / 1024:.1f} KB)")
                return str(dest)
            else:
                print(f"  Checksum mismatch for {label}, re-downloading...")
        else:
            print(f"  {label} already exists, skipping (use --force to re-download)")
            return str(dest)

    # Download
    success = download_file(url, str(dest), desc=label)
    if not success:
        print(f"\n  Failed to download {label}")
        return None

    # Verify checksum
    expected_sha = meta.get("sha256")
    if expected_sha and not no_verify:
        if not verify_checksum(str(dest), expected_sha):
            print(f"  Checksum verification failed for {label}!")
            print(f"    File may be corrupted. Re-run with --force to re-download.")
            return None

    return str(dest)


def main():
    parser = argparse.ArgumentParser(
        description="Download Josi v6 model + knowledge bundle from the Hetzner training server"
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
    print("  MiValta Josi v6 — Model + Knowledge Download")
    print("  Single model: Qwen3 (dual-mode, router-controlled)")
    print("=" * 60)
    print(f"\n  Server: {base_url}")
    print(f"  Output: {output_dir}")

    # Fetch manifest
    manifest = fetch_manifest(base_url)
    if manifest:
        print(f"  Version: {manifest.get('version', 'unknown')}")
        print(f"  Published: {manifest.get('published', 'unknown')}")

    model_meta = (manifest or {}).get("model", DEFAULT_MODEL)
    knowledge_meta = (manifest or {}).get("knowledge", DEFAULT_KNOWLEDGE)

    # ── 1. Download model GGUF ──────────────────────────────────
    print(f"\n{'─'*60}")
    print("  1/2  Model (GGUF)")
    print(f"{'─'*60}")

    model_path = download_and_verify(
        model_meta, base_url, output_dir,
        DEFAULT_MODEL["file"], "model",
        args.force, args.no_verify,
    )

    # ── 2. Download knowledge.json ──────────────────────────────
    print(f"\n{'─'*60}")
    print("  2/2  Knowledge cards (context for coaching)")
    print(f"{'─'*60}")

    knowledge_path = download_and_verify(
        knowledge_meta, base_url, output_dir,
        DEFAULT_KNOWLEDGE["file"], "knowledge",
        args.force, args.no_verify,
    )

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    ok = model_path and knowledge_path
    if ok:
        print("  Bundle Complete!")
    else:
        print("  Download finished with errors")
    print(f"{'='*60}")

    if model_path:
        size_gb = os.path.getsize(model_path) / (1024**3)
        print(f"\n  Model:     {model_path} ({size_gb:.2f} GB)")

    if knowledge_path:
        size_kb = os.path.getsize(knowledge_path) / 1024
        # Show entry count
        try:
            with open(knowledge_path) as f:
                kdata = json.load(f)
            entries = kdata.get("total_entries", len(kdata.get("entries", [])))
            print(f"  Knowledge: {knowledge_path} ({size_kb:.0f} KB, {entries} cards)")
        except Exception:
            print(f"  Knowledge: {knowledge_path} ({size_kb:.0f} KB)")

    if ok:
        print(f"\n  Both files go together — the app needs BOTH to run Josi.")
        print(f"  The model generates text, the knowledge cards provide coaching context.")

    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
