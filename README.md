# Quant13

## Overview
Quant13 is a hybrid multi-agent research and trading framework for equity options. The system orchestrates domain-specific analyst agents, a debate & decision layer, and a trader with risk management overlays. Early phases of the roadmap use deterministic mocks for complex external services (Knowledge Graph, Deep Learning Oracle) so the core framework can be developed and tested end-to-end before live integrations are introduced.

## Architecture Snapshot
- **Data Pipeline**: Pulls OHLCV, options chain, news, and company overview data via `yfinance`, then enriches it with return features.
- **Analyst Agents** (Pydantic-backed reports):
  - *VolatilityModelingAgent* — computes realized volatility, IV rank, skew, and term structure; forecasts are mocked as configurable placeholders.
  - *SentimentAgent* — scores headlines with simple keyword heuristics and triggers mocked Knowledge Graph alerts for extreme sentiment.
  - *TechnicalAnalyst* — calculates EMA/RSI/MACD indicators, extracts key levels, and appends a mocked DL Oracle forecast.
  - *FundamentalAnalyst* — surfaces basic valuation metrics from Yahoo fundamentals and combines them with a mocked Knowledge Graph SWOT summary.
- **Debate & Decision Layer**: Bullish and Bearish researcher agents argue via the OpenRouter LLM API, with a Moderator agent synthesizing the trade thesis.
- **Trader Agent**: Converts the thesis and volatility context into a concrete option strategy proposal, again through the LLM (with deterministic fallback when no API key is provided).
- **Risk Management Team**: Applies lightweight safe/neutral/risky overlays to the proposed trade.

All agent outputs conform to Pydantic models defined in `src/models/schemas.py`, ensuring type-safe message passing throughout the pipeline.

## Project Structure
```
Quant13/
├── config/
│   └── config.yaml          # Prompts, defaults, and mock payloads
├── src/
│   ├── agents/              # Agent implementations (base, analysts, debate, trader, risk)
│   ├── data/                # Data fetching & preprocessing utilities
│   ├── models/              # Pydantic schemas
│   ├── tools/               # LLM client + mocked KG/DL tools
│   ├── utils/               # Indicator helpers
│   └── orchestrator.py      # End-to-end pipeline entrypoint
├── tests/
│   └── test_agents.py       # Unit & orchestration tests
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
- Data lookback windows and news limits
- Mock outputs for Knowledge Graph and DL Oracle tools
- Risk management guidance text

## Running the Pipeline
```bash
python main.py TICKER
```
Output: a formatted JSON trade proposal driven by the orchestrator plus the underlying agent reports logged to stdout.

## Testing
```bash
.venv/bin/pytest
```
The suite covers individual agent behaviors, indicator calculations, and a fully mocked end-to-end run of the orchestrator.

## Roadmap Alignment
This repository implements Phases 1–3 of the provided specification using mocked external services. Phase 4 will replace the placeholder Knowledge Graph and DL Oracle modules with live integrations, while Phase 5 will extend the system with portfolio construction, execution, and feedback agents.