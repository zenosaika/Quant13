from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from src.orchestrator import run_pipeline


def main() -> None:
    if len(sys.argv) < 2:
        print("Ticker symbol is required", file=sys.stderr)
        sys.exit(1)

    ticker = sys.argv[1].strip().upper()

    try:
        results = run_pipeline(ticker)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    trade = results["trade_proposal"].model_dump()
    thesis = results["trade_thesis"].model_dump()
    risk = results["risk_assessment"].model_dump()
    volatility = results["volatility_report"].model_dump()
    sentiment = results["sentiment_report"].model_dump()
    technical_report = results["technical_report"]
    fundamental_report = results["fundamental_report"]

    technical_payload = {
        "llm_report": technical_report.llm_report,
        "derived_signals": getattr(technical_report.indicators, "derived_signals", []),
        "key_levels": getattr(technical_report.indicators, "key_levels", {}),
        "recent_patterns": getattr(technical_report.indicators, "recent_candlestick_patterns", []),
    }

    qualitative_summary = fundamental_report.qualitative_summary
    if hasattr(qualitative_summary, "model_dump"):
        qualitative_summary = qualitative_summary.model_dump()

    fundamental_payload = {
        "llm_synthesis": fundamental_report.llm_synthesis,
        "financial_ratios": fundamental_report.financial_ratios.model_dump(),
        "financial_trends": [trend.model_dump() for trend in fundamental_report.financial_trends[:3]],
        "qualitative_summary": qualitative_summary,
        "business_overview": fundamental_report.business_overview,
    }

    payload = {
        "ticker": ticker,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trade": trade,
        "thesis": {
            "summary": thesis.get("summary"),
            "conviction_level": thesis.get("conviction_level"),
            "key_evidence": thesis.get("key_evidence", []),
        },
        "risk": risk,
        "results_path": results.get("results_path"),
        "agents": {
            "volatility": volatility,
            "sentiment": sentiment,
            "technical": technical_payload,
            "fundamental": fundamental_payload,
        },
    }

    sys.stdout.write(json.dumps(payload))


if __name__ == "__main__":
    main()
