#!/usr/bin/env python3
"""Launch the association workbench."""

import argparse

from tracker.server import serve

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Windowed multi-target association workbench")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    serve(args.host, args.port)
