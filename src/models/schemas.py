from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


class VolatilityReport(BaseModel):
    agent: str = Field(default="VolatilityModelingAgent")
    ticker: str
    iv_rank: float
    volatility_forecast: str
    skew_analysis: str
    term_structure: str


class ArticleSentiment(BaseModel):
    title: str
    publisher: Optional[str] = None
    link: Optional[str] = None
    published_at: Optional[str] = None
    summary: Optional[str] = None
    sentiment_score: float
    rationale: Optional[str] = None


class SentimentReport(BaseModel):
    agent: str = Field(default="SentimentAgent")
    ticker: str
    overall_sentiment_score: float
    overall_summary: str
    articles: List[ArticleSentiment]


class TechnicalIndicators(BaseModel):
    latest_close: Optional[float] = None
    price_date: Optional[str] = None
    SMA_50: Optional[Dict[str, Any]] = None
    SMA_200: Optional[Dict[str, Any]] = None
    EMA_20: Optional[Dict[str, Any]] = None
    Bollinger_Bands: Dict[str, Any] = Field(default_factory=dict)
    MACD_Signal: Dict[str, Any] = Field(default_factory=dict)
    RSI: Dict[str, Any] = Field(default_factory=dict)
    Supertrend_Signal: Dict[str, Any] = Field(default_factory=dict)
    OBV_Trend: Dict[str, Any] = Field(default_factory=dict)
    recent_candlestick_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    derived_signals: List[str] = Field(default_factory=list)
    key_levels: Dict[str, Optional[float]] = Field(default_factory=dict)


class TechnicalReport(BaseModel):
    agent: str = Field(default="TechnicalAnalyst")
    ticker: str
    indicators: TechnicalIndicators
    llm_report: Dict[str, Any] = Field(default_factory=dict)
    llm_raw: Optional[str] = None


class FinancialRatios(BaseModel):
    pe_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None


class FinancialTrend(BaseModel):
    metric: str
    values: Dict[str, Optional[float]] = Field(default_factory=dict)
    trend_direction: Optional[str] = None
    compound_growth_rate: Optional[float] = None


class QualitativeSummary(BaseModel):
    mdna_summary: Optional[Dict[str, Any]] = None
    risk_factors: List[Dict[str, Any]] = Field(default_factory=list)


class FundamentalReport(BaseModel):
    agent: str = Field(default="FundamentalAnalyst")
    ticker: str
    generated_at: str
    data_source: str
    business_overview: Dict[str, Any] = Field(default_factory=dict)
    financial_ratios: FinancialRatios
    financial_trends: List[FinancialTrend]
    qualitative_summary: QualitativeSummary
    llm_synthesis: Dict[str, Any] = Field(default_factory=dict)


class DebateArgument(BaseModel):
    stance: str
    argument: str


class TradeThesis(BaseModel):
    winning_argument: str
    conviction_level: str
    summary: str
    key_evidence: List[str]


class TradeLeg(BaseModel):
    contract_symbol: str
    type: str
    action: str
    strike_price: float
    expiration_date: str
    quantity: int = 1
    key_greeks_at_selection: Dict[str, Optional[float]] = Field(default_factory=dict)


class TradeProposal(BaseModel):
    agent: str = Field(default="TraderAgent")
    strategy_name: str
    action: str
    quantity: int
    trade_legs: List[TradeLeg]
    notes: Optional[str] = None
    conviction_level: Optional[str] = None
    generation_status: Literal["generated", "failed"] = "generated"
    max_risk: Optional[float] = None
    max_reward: Optional[float] = None
    net_premium: Optional[float] = None


class RiskAdjustment(BaseModel):
    profile: str
    recommendation: str


class RiskAssessment(BaseModel):
    adjustments: List[RiskAdjustment]
    final_recommendation: str


class BacktestReport(BaseModel):
    agent: str = Field(default="BacktesterAgent")
    ticker: str
    strategy_type: str
    win_rate: float
    simulated_trades: int
    average_profit_pct: float
    summary: str
