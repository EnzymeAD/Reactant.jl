#!/usr/bin/env python3
"""
Serve XLA profiler traces for viewing in Perfetto UI.

Starts an HTTP server with CORS headers, finds trace files, and prints
Perfetto URLs. Works for both NumPyro and Reactant traces.

Usage:
    python serve_traces.py [--port PORT] [--dir DIR]

On a remote machine, port-forward first:
    ssh -L 9001:localhost:9001 <host>
Then open the printed Perfetto URL in your browser.
"""

import argparse
import glob
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


class CORSHandler(SimpleHTTPRequestHandler):
    """HTTP handler with CORS headers (required by Perfetto UI)."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress request logging noise
        pass


def find_traces(base_dir):
    """Find all .trace.json.gz files, return sorted by mtime (newest first)."""
    pattern = os.path.join(base_dir, "**", "*.trace.json.gz")
    files = glob.glob(pattern, recursive=True)
    return sorted(files, key=os.path.getmtime, reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Serve XLA traces for Perfetto UI")
    parser.add_argument("--port", type=int, default=9001, help="HTTP server port")
    parser.add_argument(
        "--dir",
        default=None,
        help="Trace directory to serve (default: latest timestamped output)",
    )
    args = parser.parse_args()

    if args.dir is not None:
        trace_dir = os.path.abspath(args.dir)
    else:
        # Find latest timestamped output dir containing traces
        outputs_root = os.path.join(os.path.dirname(__file__), "outputs")
        trace_dir = None
        if os.path.isdir(outputs_root):
            candidates = sorted(
                [d for d in os.listdir(outputs_root)
                 if os.path.isdir(os.path.join(outputs_root, d, "traces"))],
                reverse=True,
            )
            if candidates:
                trace_dir = os.path.join(outputs_root, candidates[0], "traces")
        if trace_dir is None:
            trace_dir = os.path.join(outputs_root, "traces")
        trace_dir = os.path.abspath(trace_dir)
    if not os.path.isdir(trace_dir):
        print(f"Error: trace directory not found: {trace_dir}")
        print("Run the benchmark with --profile first.")
        sys.exit(1)

    traces = find_traces(trace_dir)
    if not traces:
        print(f"No .trace.json.gz files found in {trace_dir}")
        print("Run the benchmark with --profile first.")
        sys.exit(1)

    # Group by framework (numpyro / reactant)
    grouped = {}
    for path in traces:
        rel = os.path.relpath(path, trace_dir)
        framework = rel.split(os.sep)[0] if os.sep in rel else "unknown"
        grouped.setdefault(framework, []).append(path)

    # Start server from trace_dir so paths are relative
    os.chdir(trace_dir)

    print(f"Serving traces from: {trace_dir}")
    print(f"HTTP server: http://127.0.0.1:{args.port}/")
    print()
    print("=== Perfetto URLs ===")
    for framework, files in sorted(grouped.items()):
        latest = files[0]  # newest first
        rel_path = os.path.relpath(latest, trace_dir)
        url = f"https://ui.perfetto.dev/#!/?url=http://127.0.0.1:{args.port}/{rel_path}"
        print(f"  {framework} (latest): {url}")
    print()
    print("On a remote machine, port-forward first:")
    print(f"  ssh -L {args.port}:localhost:{args.port} <host>")
    print()
    print("Press Ctrl+C to stop.")
    print()

    server = HTTPServer(("0.0.0.0", args.port), CORSHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
