#!/usr/bin/env python3
"""
Standalone Discord sender for CLI integration.
Takes result directory path and sends to Discord.
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.integrations.discord_webhook import send_to_discord


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: discord_sender.py <ticker> <results_path>", file=sys.stderr)
        sys.exit(1)

    ticker = sys.argv[1].upper()
    results_path = Path(sys.argv[2])

    if not results_path.exists():
        print(f"Error: Results path does not exist: {results_path}", file=sys.stderr)
        sys.exit(1)

    # Send to Discord
    success = send_to_discord(ticker, results_path)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
