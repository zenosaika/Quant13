# Quant13

## Overview
Quant13 is a hybrid multi-agent research and trading framework for equity options. The system orchestrates domain-specific analyst agents, a debate & decision layer, and a trader with risk management overlays. The current implementation is fully data-driven: agents source live market data via `yfinance`, compute quantitative signals locally, and use OpenRouter LLM calls strictly for synthesis.

## Architecture Snapshot
- **Data Pipeline**: Pulls OHLCV, options chain, news, company fundamentals, filings, and financial statements via `yfinance`, then enriches them with return features and indicator bundles.
- **Analyst Agents** (Pydantic-backed reports):
  - *VolatilityModelingAgent* — computes realized volatility, IV rank, skew, term structure, and classifies the realized vol trend.
  - *SentimentAgent* — normalizes recent headlines/news, forwards them to the LLM for scoring, and aggregates a narrative-aware sentiment report.
  - *TechnicalAnalyst* — generates an expanded indicator bundle (SMAs, EMA, Bollinger Bands, MACD, RSI, Supertrend, OBV, candlestick patterns), then prompts the LLM for synthesis of bias/levels.
  - *FundamentalAnalyst* — caches analyses per ticker, calculates key ratios and multi-year trends, scrapes the latest SEC filings for MD&A/Risk sections, summarizes them with the LLM, and produces a full SWOT/thesis.
- **Debate & Decision Layer**: Bullish and Bearish researcher agents argue via the OpenRouter LLM API, with a Moderator agent synthesizing the trade thesis.
- **Trader Agent**: Consumes the trade thesis, volatility report, and full options chain to return executable option legs (strategy, strikes, greeks) via the LLM.
- **Risk Management Team**: Applies lightweight safe/neutral/risky overlays to the proposed trade.

All agent outputs conform to Pydantic models defined in `src/models/schemas.py`, ensuring type-safe message passing throughout the pipeline.

## Project Structure
```
Quant13/
├── config/
│   └── config.yaml          # Prompts, defaults, and data/agent configuration
├── src/
│   ├── agents/              # Agent implementations (base, analysts, debate, trader, risk)
│   ├── data/                # Data fetching & preprocessing utilities
│   ├── models/              # Pydantic schemas
│   ├── tools/               # LLM client utilities
│   ├── utils/               # Indicator helpers
│   └── orchestrator.py      # End-to-end pipeline entrypoint
├── tests/
│   └── test_agents.py       # Unit & orchestration tests
├── cache/                   # Fundamental analysis cache (auto-created)
├── results/                 # Timestamped agent and trade outputs (auto-created)
├── requirements.txt
└── main.py                  # CLI runner
```

## Prerequisites
- Python 3.10+ (repo tested with 3.14 via virtual environment)
- [OpenRouter](https://openrouter.ai) API key (optional; enables live debate/trader responses)

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure your OpenRouter credentials by exporting the API key expected in `config/config.yaml`:
```bash
export OPENROUTER_API_KEY="your_key_here"
```
If the key is omitted, deterministic fallback responses keep the pipeline operational for development and testing.

## Configuration
Modify `config/config.yaml` to tune:
- LLM model, temperature, and prompt templates
- Data lookback windows, indicator windows, and news limits
- Volatility lookbacks and sentiment/time filters
- Risk management guidance text

## Running the Pipeline
```bash
python main.py TICKER
```
Output: a formatted JSON trade proposal printed to stdout, plus timestamped JSON reports for each agent, thesis, trader decision, and risk overlay written to `results/<TICKER>_<YYYYMMDD_HHMMSS>/`.

## Project 13 Immersive CLI
An Ink-powered terminal experience ships alongside the Python entry point. It renders the full Project 13 agent collective with persona colors, live spinners, gauges, and a splash intro.

1. Install Node.js ≥ 18 (Node 24 is pre-installed in the dev environment).
2. From the repo root:
   ```bash
   cd cli
   npm install
   ```
3. Launch the console interactively:
   ```bash
   npm start
   ```
   Or run non-interactively (useful for scripts/CI):
   ```bash
   printf "AAPL\n" | npm start
   ```

Highlights:
- Animated “Project 13” splash art and persistent status header (ticker focus, working directory).
- Persona-specific frames for Volatility, Sentiment, Technical, Fundamental, Debate, Trader, Risk, and Artifact courier agents.
- Rich metadata (IV rank gauges, sentiment arcs, thesis evidence, option-leg breakdowns) streamed directly from the Python pipeline via `src/cli_bridge.py`.
- Non-interactive mode auto-exits after rendering the full report, making it drop-in for automation.

## Testing
```bash
.venv/bin/pytest
```
The suite covers agent primitives, technical indicator generation, and an end-to-end orchestrator run under deterministic LLM mocks.

## Utilities
- `data_test.py` offers a quick sanity check for raw `yfinance` structures (OHLCV, options chains, news, info, filings) before extending agent logic.

## Roadmap Alignment
This repository now reflects the data-first evolution of the specification—mock services have been removed in favor of real market data, resilient scraping, and LLM synthesis. Future phases can build atop this foundation to add portfolio construction, execution, and feedback agents.