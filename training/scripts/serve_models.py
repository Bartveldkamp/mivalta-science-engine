#!/usr/bin/env python3
"""
MiValta Josi v4 — Simple Model File Server

Serves GGUF model files over HTTP so your developer can download them directly.
Runs as a background process that persists after you close your SSH session.

Usage:
    # Start serving (auto-detects GGUF files)
    python serve_models.py

    # Specify port (default: 8079)
    python serve_models.py --port 9000

    # Serve from a specific directory
    python serve_models.py --model-dir /path/to/gguf/files

    # Run in background (persists after SSH disconnect)
    python serve_models.py --daemon

    # Stop background server
    python serve_models.py --stop

    # Show status
    python serve_models.py --status

Download URLs will be:
    http://<your-server-ip>:8079/josi-v4-interpreter-q4_k_m.gguf
    http://<your-server-ip>:8079/josi-v4-explainer-q4_k_m.gguf
"""

import argparse
import http.server
import json
import os
import signal
import socket
import subprocess
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "gguf"
TRAINING_MODEL_DIR = SCRIPT_DIR.parent / "models" / "gguf"
PID_FILE = Path("/tmp/mivalta-model-server.pid")
DEFAULT_PORT = 8079


def find_gguf_dir() -> Path:
    """Find directory containing GGUF files."""
    for d in [DEFAULT_MODEL_DIR, TRAINING_MODEL_DIR]:
        if d.exists() and list(d.glob("*.gguf")):
            return d
    return DEFAULT_MODEL_DIR


def get_server_ips() -> list[str]:
    """Get all non-loopback IP addresses of this machine."""
    ips = []
    try:
        # Get hostname-based IP
        hostname = socket.gethostname()
        ips.append(socket.gethostbyname(hostname))
    except Exception:
        pass

    # Try to get the public IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass

    return list(dict.fromkeys(ip for ip in ips if ip != "127.0.0.1"))


class ModelFileHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that only serves .gguf and .json files."""

    def __init__(self, *args, model_dir: str, **kwargs):
        super().__init__(*args, directory=model_dir, **kwargs)

    def do_GET(self):
        # Serve index at root
        if self.path == "/" or self.path == "/index.html":
            self.send_index()
            return

        # Only serve .gguf and .json files
        clean_path = self.path.lstrip("/").split("?")[0]
        if not (clean_path.endswith(".gguf") or clean_path.endswith(".json")):
            self.send_error(404, "Not Found")
            return

        super().do_GET()

    def send_index(self):
        """Generate a simple HTML index page listing available models."""
        model_dir = Path(self.directory)
        files = sorted(model_dir.glob("*.gguf"))

        html = "<html><head><title>MiValta Josi v4 Models</title></head><body>\n"
        html += "<h1>MiValta Josi v4 — GGUF Models</h1>\n"
        html += "<table border='1' cellpadding='8'>\n"
        html += "<tr><th>File</th><th>Size</th><th>Download</th></tr>\n"

        for f in files:
            size_gb = f.stat().st_size / (1024**3)
            html += f"<tr><td>{f.name}</td><td>{size_gb:.2f} GB</td>"
            html += f"<td><a href='/{f.name}'>Download</a></td></tr>\n"

        if not files:
            html += "<tr><td colspan='3'>No GGUF files found</td></tr>\n"

        html += "</table>\n"
        html += f"<p><small>Served at {datetime.now().strftime('%Y-%m-%d %H:%M')}</small></p>\n"
        html += "</body></html>"

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html.encode())

    def log_message(self, format, *args):
        """Custom log format."""
        sys.stderr.write(
            f"[{datetime.now().strftime('%H:%M:%S')}] {self.address_string()} - {format % args}\n"
        )


def start_server(model_dir: str, port: int):
    """Start the HTTP file server."""
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"  ERROR: Directory not found: {model_dir}")
        sys.exit(1)

    gguf_files = list(model_path.glob("*.gguf"))
    if not gguf_files:
        print(f"  WARNING: No .gguf files found in {model_dir}")
        print(f"  The server will start but there's nothing to download yet.")

    ips = get_server_ips()

    print("=" * 60)
    print("  MiValta Josi v4 — Model File Server")
    print("=" * 60)
    print(f"\n  Serving from: {model_dir}")
    print(f"  Port: {port}")
    print(f"\n  Available files:")
    for f in gguf_files:
        size_gb = f.stat().st_size / (1024**3)
        print(f"    - {f.name}  ({size_gb:.2f} GB)")

    print(f"\n  Download URLs for your developer:")
    for ip in ips:
        for f in gguf_files:
            print(f"    http://{ip}:{port}/{f.name}")

    if not ips:
        print(f"    http://localhost:{port}/")
        print(f"  (Could not detect server IP — use your server's public IP)")

    print(f"\n  Stop with: python {Path(__file__).name} --stop")
    print(f"  Or: Ctrl+C")
    print()

    handler = partial(ModelFileHandler, model_dir=str(model_dir))
    server = http.server.HTTPServer(("0.0.0.0", port), handler)

    # Save PID
    PID_FILE.write_text(str(os.getpid()))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
    finally:
        server.server_close()
        if PID_FILE.exists():
            PID_FILE.unlink()


def start_daemon(model_dir: str, port: int):
    """Start server as a background daemon process."""
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"  Server already running (PID {pid})")
            print(f"  Stop with: python {Path(__file__).name} --stop")
            return
        except ProcessLookupError:
            PID_FILE.unlink()

    log_file = Path("/tmp/mivalta-model-server.log")
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--model-dir", str(model_dir),
        "--port", str(port),
    ]

    print(f"  Starting server in background...")
    print(f"  Log: {log_file}")

    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    print(f"  Server started (PID {proc.pid})")
    print(f"\n  Stop with: python {Path(__file__).name} --stop")

    # Wait briefly and check it started OK
    import time
    time.sleep(1)
    if proc.poll() is not None:
        print(f"  ERROR: Server exited immediately. Check {log_file}")
        return

    ips = get_server_ips()
    model_path = Path(model_dir)
    gguf_files = list(model_path.glob("*.gguf"))

    print(f"\n  Download URLs for your developer:")
    for ip in ips:
        for f in gguf_files:
            print(f"    http://{ip}:{port}/{f.name}")


def stop_server():
    """Stop the background server."""
    if not PID_FILE.exists():
        print("  No server running (no PID file)")
        return

    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"  Server stopped (PID {pid})")
    except ProcessLookupError:
        print(f"  Server was not running (stale PID {pid})")

    PID_FILE.unlink()


def show_status():
    """Show server status."""
    if not PID_FILE.exists():
        print("  Server: not running")
        return

    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, 0)
        print(f"  Server: running (PID {pid})")
    except ProcessLookupError:
        print(f"  Server: not running (stale PID {pid})")
        PID_FILE.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Serve GGUF model files over HTTP for developer download"
    )
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory containing GGUF files")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP port (default: {DEFAULT_PORT})")
    parser.add_argument("--daemon", action="store_true",
                        help="Run in background (persists after SSH disconnect)")
    parser.add_argument("--stop", action="store_true",
                        help="Stop background server")
    parser.add_argument("--status", action="store_true",
                        help="Show server status")

    args = parser.parse_args()

    if args.stop:
        stop_server()
        return

    if args.status:
        show_status()
        return

    model_dir = args.model_dir or str(find_gguf_dir())

    if args.daemon:
        start_daemon(model_dir, args.port)
    else:
        start_server(model_dir, args.port)


if __name__ == "__main__":
    main()
