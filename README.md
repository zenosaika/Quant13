# 🎯 Quant13

**A Hybrid Multi-Agent Options Trading System**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Quant13 is an intelligent options trading analysis framework that combines the reasoning capabilities of Large Language Models (LLMs) with the precision of quantitative finance models. Built with a hybrid architecture, it addresses the critical challenge of AI hallucination in financial systems by separating qualitative analysis (handled by LLMs) from quantitative calculations (handled by deterministic algorithms).

---

## ✨ Key Features

### 🤖 **Multi-Agent Intelligence**
- **7 Specialized AI Agents** working in concert:
  - **Volatility Analyst**: IV rank, term structure, skew analysis
  - **Sentiment Analyst**: Multi-source news sentiment with lexicon scoring
  - **Technical Analyst**: RSI, MACD, Supertrend, Bollinger Bands
  - **Fundamental Analyst**: Financial ratios, MD&A analysis, risk factors
  - **Debate Team**: Bull vs Bear with Moderator consensus
  - **Risk Manager**: Stress testing and position sizing
  - **Fund Manager**: Final go/no-go decisions

### 📊 **Quantitative Rigor**
- **Black-Scholes Greeks** calculation for all options
- **Systematic strategy selection** based on conviction + IV regime
- **Deterministic risk metrics**: Max risk, max reward, net premium
- **Delta-based strike selection** for mathematical precision
- **Thesis-strategy alignment validation** to prevent mismatches

### 🎨 **Professional Reporting**
- **Dual-language PDF reports** (English & Thai) with strategy explanations
- **Discord webhook integration** for instant notifications
- **Interactive CLI** built with React & Ink
- **Comprehensive audit trails** for every decision

### 🛡️ **Built for Reliability**
- **Hybrid architecture** minimizes AI hallucination risk
- **Pydantic validation** ensures data integrity
- **Configurable validation gates** with strict/warning modes
- **Explainable AI**: Every decision has traceable reasoning

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for CLI interface)
- **OpenRouter API key** ([Get one here](https://openrouter.ai/))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Quant13.git
cd Quant13
```

2. **Set up Python environment**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenRouter API key
# OPENROUTER_API_KEY=your_key_here
```

4. **Install CLI dependencies (optional)**
```bash
cd cli
npm install
cd ..
```

### Basic Usage

**Analyze a ticker:**
```bash
python main.py TSLA
```

**With options:**
```bash
# Skip PDF generation
python main.py NVDA --no-pdf

# Enable verbose logging
python main.py AAPL -v

# Send to Discord
python main.py MSFT --discord

# Generate dual-language reports
python main.py GOOGL --dual-language
```

**Use interactive CLI:**
```bash
cd cli
npm start
```

---

## 🏗️ System Architecture

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Data Collection                                         │
│     • OHLCV data (yfinance)                                 │
│     • Options chain + Greeks enrichment (Black-Scholes)     │
│     • News & fundamentals                                   │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Parallel Analyst Phase (Concurrent Execution)           │
│     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│     │ Volatility  │  │  Sentiment  │  │  Technical  │       │
│     └─────────────┘  └─────────────┘  └─────────────┘       │
│     ┌─────────────┐                                         │
│     │Fundamental  │                                         │
│     └─────────────┘                                         │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Debate & Thesis Formation                               │
│     Bull Researcher ⚔️ Bear Researcher → Moderator          |
│     Output: Direction + Conviction Level                    │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Systematic Trade Construction                           │
│     • Strategy scoring algorithm (Direction + IV + Conv.)   │
│     • Delta-based strike selection                          │
│     • Multi-leg strategy assembly                           │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Validation & Risk Calculation                           │
│     • Thesis-strategy alignment check                       │
│     • Black-Scholes pricing for spreads                     │
│     • Max risk/reward computation                           │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Final Review & Decision                                 │
│     Risk Manager → Fund Manager → Execute/Reject            │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  7. Reporting & Notifications                               │
│     • JSON reports saved to results/                        │
│     • PDF generation (English/Thai)                         │
│     • Discord webhook notification                          │
└─────────────────────────────────────────────────────────────┘
```

### Core Design Principles

**🎭 Hybrid Architecture**
- **LLMs handle**: Qualitative reasoning, sentiment analysis, thesis formation
- **Code handles**: Mathematical calculations, strike selection, risk metrics
- **Why**: Eliminates AI hallucination in critical numerical operations

**🔍 Systematic Over Heuristic**
- Default uses `SystematicTraderAgent` with rule-based strategy selection
- Strategy library with deterministic scoring: `Direction (40%) + IV Regime (30%) + Conviction (20%)`
- Delta-based strike targeting (e.g., Delta 0.30 for ~70% OTM probability)
- LLM-based trader available as fallback with `--no-systematic-trader`

**✅ Multi-Layer Validation**
- **Phase 1**: Thesis-strategy alignment (prevents bearish thesis + bullish strategy)
- **Phase 2**: Risk metrics sanity checks
- **Phase 3**: Manager-level go/no-go decision
- Configurable strict/warning modes in `config/config.yaml`

---

## 📁 Project Structure

```
Quant13/
├── main.py                      # Main entry point
├── generate_report.py           # Standalone PDF generator
├── config/
│   └── config.yaml             # Agent prompts & parameters
├── src/
│   ├── agents/                 # AI agent implementations
│   │   ├── volatility.py
│   │   ├── sentiment_v2.py     # Enhanced multi-source sentiment
│   │   ├── technical.py
│   │   ├── fundamental.py
│   │   ├── debate.py
│   │   ├── trader_v2.py        # Systematic trader
│   │   ├── risk.py
│   │   └── manager.py
│   ├── data/                   # Data fetchers
│   │   ├── fetcher.py          # yfinance integration
│   │   ├── sec.py              # SEC filings
│   │   └── sentiment_lexicon.py
│   ├── pricing/                # Quantitative finance models
│   │   ├── black_scholes.py
│   │   ├── greeks_engine.py
│   │   └── risk_free_rate.py
│   ├── strategies/             # Strategy library
│   │   ├── strategy_library.py
│   │   ├── strategy_selector.py
│   │   └── strike_selector.py
│   ├── validation/             # Validation framework
│   │   └── thesis_validator.py
│   ├── utils/                  # Utilities
│   │   ├── indicators.py       # Technical indicators
│   │   └── risk.py            # Risk calculations
│   ├── orchestrator_v2.py      # Main pipeline orchestrator
│   └── config.py
├── cli/                        # TypeScript/React CLI
│   └── src/
│       └── cli.tsx
├── templates/                  # PDF report templates
│   └── report_template.html
├── results/                    # Analysis outputs (gitignored)
└── cache/                      # Data cache (gitignored)
```

---

## 🎓 How It Works

### Example: NVDA Analysis

Let's walk through a real analysis:

**1. Data Collection**
```bash
python main.py NVDA -v
```
- Fetches 120 days of OHLCV data
- Downloads options chain for all expirations
- Enriches each option with Black-Scholes Greeks (Delta, Gamma, Theta, Vega, Rho)
- Pulls latest news articles and SEC filings

**2. Analyst Reports**
- **Volatility Agent**: "IV Rank = 75 (High) → Term structure inverted → Favor credit strategies"
- **Sentiment Agent**: "10 articles analyzed, Avg sentiment = +0.6 (Bullish) → Positive earnings reaction"
- **Technical Agent**: "RSI = 65, MACD bullish crossover → Uptrend confirmed, resistance at $950"
- **Fundamental Agent**: "P/E stretched but strong revenue growth → Quality company, valuation concerns"

**3. Debate Phase**
- **Bull**: "Strong technicals + positive sentiment support upside to $950"
- **Bear**: "High IV means options are expensive, risk of IV crush post-earnings"
- **Moderator**: "**Winner: Bullish** (Medium conviction) → Upside potential but manage IV risk"

**4. Strategy Selection**
```
Scoring Algorithm:
- Direction: Bullish ✓ (40 points)
- IV Regime: High (75) → Credit strategies favored (30 points)
- Conviction: Medium (15 points)

Top Strategy: Bull Put Spread (85/100 score)
Rationale: Bullish bias + High IV → Sell premium instead of buying expensive calls
```

**5. Trade Construction**
```
Selected Strategy: Bull Put Spread
- Sell 1 Put @ Strike $900 (Delta -0.30)  → Collect premium
- Buy 1 Put @ Strike $880 (Delta -0.20)   → Define max risk
- Expiration: 30 DTE
- Net Credit: $420
- Max Risk: $1,580
- Max Reward: $420
- Breakeven: $895.80
```

**6. Risk Review**
- **Risk Manager**: "Trade structure valid. Sizing recommendation: Half position (concerns: high IV, earnings proximity)"
- **Fund Manager**: "**Execute at 50% size**. Thesis-strategy aligned. Monitor for IV crush."

**7. Output**
```
Results saved to: results/NVDA_20251123_114451/
- fundamental_report.json
- technical_report.json
- volatility_report.json
- sentiment_report.json
- trade_thesis.json
- trade_decision.json
- risk_assessment.json
- NVDA_20251123_114451_report.pdf
```

---

## ⚙️ Configuration

All agent behaviors are configurable via `config/config.yaml`:

```yaml
openrouter:
  model: "google/gemini-2.5-flash-preview-09-2025"

agents:
  volatility:
    iv_rank_lookback_days: 30
  trader:
    strategy_preferences:
      high_iv_rank_threshold: 50
      low_iv_rank_threshold: 30

validation:
  enable_thesis_gate: true
  strict_mode: true  # Raise exception on mismatch vs warning only
```

**No code changes needed** for prompt tuning or parameter adjustments.

---

## 📊 Available Strategies

The system supports 12+ options strategies:

**Directional (Bullish)**
- Long Call
- Bull Call Spread
- Bull Put Spread
- Cash-Secured Put

**Directional (Bearish)**
- Long Put
- Bear Put Spread
- Bear Call Spread

**Neutral**
- Iron Condor
- Butterfly Spread
- Straddle
- Strangle

**Income Generation**
- Covered Call

Each strategy includes:
- Deterministic risk calculations
- Beginner-friendly explanations (English & Thai)
- Pros/cons analysis
- Suitable market conditions

---

## 📈 Results & Outputs

### JSON Reports

Every analysis generates structured JSON reports:

```json
{
  "ticker": "NVDA",
  "generated_at": "2025-11-23T11:44:51Z",
  "trade_proposal": {
    "strategy_name": "Bull Put Spread",
    "action": "Bullish",
    "conviction_level": "Medium",
    "max_risk": 1580.0,
    "max_reward": 420.0,
    "net_premium": 420.0,
    "trade_legs": [...]
  }
}
```

### PDF Reports

Professional multi-page reports with:
- Executive Summary
- Market Snapshot (Price, IV Rank, Technical Bias)
- Trade Proposal with risk metrics
- Detailed analyst reports
- Strategy explanations for beginners
- Risk assessments and sizing recommendations

**Dual-language support**: Generate both English and Thai versions automatically.

### Discord Notifications

Get instant alerts with:
- Trade summary
- Key metrics (conviction, IV rank, max risk/reward)
- PDF attachment (if enabled)

---

## 🧪 Testing & Development

### Run Tests
```bash
pytest
```

### Debug Single Component
```bash
# Test volatility agent only
python -c "from src.agents.volatility import VolatilityModelingAgent; ..."

# Regenerate PDF from existing analysis
python generate_report.py NVDA 20251123_114451
```

### Check Specific Results
```bash
# View trade decision
cat results/NVDA_20251123_114451/trade_decision.json | python -m json.tool

# View debate transcript
cat results/NVDA_20251123_114451/trade_thesis.json | python -m json.tool
```

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Core Logic** | Python 3.11+ | Main analysis engine |
| **LLM Inference** | OpenRouter API | Multi-model LLM access |
| **Data Validation** | Pydantic | Type-safe data models |
| **Market Data** | yfinance | OHLCV & options data |
| **Quant Finance** | SciPy, NumPy | Black-Scholes calculations |
| **Technical Analysis** | pandas, TA-Lib patterns | Indicators & signals |
| **PDF Generation** | WeasyPrint, Jinja2 | Professional reports |
| **CLI Interface** | TypeScript, React, Ink | Interactive terminal UI |
| **Notifications** | Discord Webhooks | Real-time alerts |

---

## 🔬 Research & Academic Context

Quant13 was developed as part of academic research into hybrid AI systems for financial decision-making. Key research contributions:

1. **Addressing AI Hallucination in Finance**: Novel architecture separating qualitative reasoning (LLM) from quantitative precision (deterministic code)

2. **Multi-Agent Debate Mechanism**: Bull vs Bear debate reduces single-model bias and improves decision quality

3. **Explainable AI for Trading**: Complete audit trails showing reasoning behind every recommendation

4. **Systematic Strategy Selection**: Rule-based approach combining conviction, volatility regime, and direction

---

## 🚧 Limitations & Future Work

### Current Limitations

- **Data latency**: yfinance has ~15-20 min delay (suitable for swing trading, not day trading)
- **Historical options data**: Limited availability requires synthetic data generation for backtesting
- **API rate limits**: Aggressive data fetching may trigger rate limits

### Planned Improvements

**Phase 1: Deep Learning Integration**
- IV prediction models for better entry timing
- Optimal strike selection via reinforcement learning

**Phase 2: Knowledge Graphs**
- Supply chain relationship mapping
- Sector correlation analysis

**Phase 3: Portfolio Management**
- Multi-ticker portfolio optimization
- Greek-neutral hedging strategies

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## ⚠️ Disclaimer

**Quant13 is an academic research project and educational tool.**

- **Not financial advice**: This software does not provide investment recommendations
- **No guarantees**: Past performance does not guarantee future results
- **Use at own risk**: Trading options involves substantial risk of loss
- **For research only**: Not intended for production trading without extensive validation

Options trading can result in the loss of your entire investment. Always consult a licensed financial advisor before making investment decisions.

---

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/Quant13/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Quant13/discussions)

---

## 🙏 Acknowledgments

Built with:
- OpenRouter for LLM access
- Yahoo Finance for market data
- The open-source Python & Node.js communities
- Academic advisors and peer reviewers

---

<div align="center">

**Made with ❤️ for quantitative finance research**

⭐ Star this repo if you find it useful!

</div>
