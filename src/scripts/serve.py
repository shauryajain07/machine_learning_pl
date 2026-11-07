"""Run the MatchSense web app.

    ./.venv/bin/python -m src.scripts.serve            # live mode on :8000
    ./.venv/bin/python -m src.scripts.serve --demo     # offline demo on :8001
"""
from __future__ import annotations

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Offline demo with recorded/simulated data")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.demo:
        os.environ["MATCHSENSE_MODE"] = "demo"
    import uvicorn

    uvicorn.run("src.matchsense.api:app", host=args.host, port=args.port or (8001 if args.demo else 8000))


if __name__ == "__main__":
    main()
