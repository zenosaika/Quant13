from __future__ import annotations

import argparse
import json

from src.orchestrator import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hybrid multi-agent options trading framework.")
    parser.add_argument("ticker", help="Ticker symbol to analyze")
    args = parser.parse_args()

    results = run_pipeline(args.ticker.upper())
    trade = results["trade_proposal"].model_dump()
    print("Final Trade Proposal:\n" + json.dumps(trade, indent=2))

    pdf_path = results.get("report_pdf")
    if pdf_path:
        print(f"\nPDF report generated at: {pdf_path}")
    elif results.get("report_pdf_error"):
        print(f"\nPDF report generation failed: {results['report_pdf_error']}")


if __name__ == "__main__":
    main()
