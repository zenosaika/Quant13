from __future__ import annotations

import argparse
import json
import logging

# Use enhanced orchestrator with all improvements
from src.orchestrator_v2 import run_pipeline_v2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the enhanced multi-agent options trading framework with all improvements."
    )
    parser.add_argument("ticker", help="Ticker symbol to analyze")
    parser.add_argument(
        "--no-enhanced-sentiment",
        action="store_true",
        help="Use original sentiment agent instead of enhanced multi-source version"
    )
    parser.add_argument(
        "--no-systematic-trader",
        action="store_true",
        help="Use original LLM-based trader instead of systematic rule-based version"
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF generation"
    )
    parser.add_argument(
        "--dual-language",
        action="store_true",
        help="Generate both English and Thai PDFs (experimental)"
    )
    parser.add_argument(
        "--discord",
        action="store_true",
        help="Send results to Discord via webhook (requires DISCORD_WEBHOOK_URL env var)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )

    # Run enhanced pipeline
    print(f"\n{'='*60}")
    print(f"QUANT13 ENHANCED OPTIONS TRADING SYSTEM")
    print(f"{'='*60}")
    print(f"Ticker: {args.ticker.upper()}")
    print(f"Enhanced Sentiment: {not args.no_enhanced_sentiment}")
    print(f"Systematic Trader: {not args.no_systematic_trader}")
    print(f"PDF Generation: {not args.no_pdf}")
    print(f"Dual Language: {args.dual_language}")
    print(f"Discord Notification: {args.discord}")
    print(f"{'='*60}\n")

    results = run_pipeline_v2(
        args.ticker.upper(),
        use_enhanced_sentiment=not args.no_enhanced_sentiment,
        use_systematic_trader=not args.no_systematic_trader,
        generate_pdf=not args.no_pdf,
        dual_language=args.dual_language,
        send_to_discord=args.discord
    )

    # Print results summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")

    trade = results["trade_proposal"].model_dump()
    print("\n📊 TRADE PROPOSAL:")
    print(f"  Strategy: {trade['strategy_name']}")
    print(f"  Action: {trade['action']}")
    print(f"  Legs: {len(trade['trade_legs'])}")

    # Display capital required / net premium
    net_premium = trade.get('net_premium')
    if net_premium is not None:
        if net_premium < 0:
            print(f"  💰 Capital Required: ${abs(net_premium):,.2f}")
        else:
            print(f"  💰 Net Credit Received: ${net_premium:,.2f}")

    # Display max risk
    if trade.get('max_risk'):
        print(f"  📉 Max Risk: ${trade['max_risk']:,.2f}")

    # Display max reward
    max_reward = trade.get('max_reward')
    if max_reward:
        print(f"  📈 Max Reward: ${max_reward:,.2f}")
    elif trade.get('max_risk'):
        # If we have risk but no reward, it's unlimited
        print(f"  📈 Max Reward: Unlimited")

    print("\n📈 MARKET ANALYSIS:")
    print(f"  Thesis: {results['trade_thesis'].winning_argument}")
    print(f"  Conviction: {results['trade_thesis'].conviction_level}")

    sentiment_score = results['sentiment_report'].overall_sentiment_score
    if sentiment_score is not None:
        print(f"  Sentiment Score: {sentiment_score:.2f}")
    else:
        print(f"  Sentiment Score: N/A")

    iv_rank = results['volatility_report'].iv_rank
    if iv_rank is not None:
        print(f"  IV Rank: {iv_rank:.1f}")
    else:
        print(f"  IV Rank: N/A")

    print(f"\n📁 RESULTS:")
    print(f"  Path: {results.get('results_path', 'N/A')}")

    pdf_path = results.get("report_pdf")
    if pdf_path:
        print(f"  PDF: {pdf_path}")
    elif results.get("report_pdf_error"):
        print(f"  PDF Error: {results['report_pdf_error']}")

    # Discord status
    if args.discord:
        if results.get("discord_sent"):
            print(f"  Discord: Sent successfully!")
        elif results.get("discord_error"):
            print(f"  Discord Error: {results['discord_error']}")
        else:
            print(f"  Discord: Not sent (check config)")

    print(f"\n{'='*60}\n")

    # Print detailed trade proposal
    print("\nDETAILED TRADE PROPOSAL:")
    print(json.dumps(trade, indent=2))

    print(f"\n✅ Analysis complete!\n")


if __name__ == "__main__":
    main()
