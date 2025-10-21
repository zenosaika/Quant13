from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class NewsHeadline(BaseModel):
    headline: str
    summary: Optional[str] = None
    link: Optional[str] = None
    published_at: Optional[str] = None
    score: Optional[float] = None


class VolatilityReport(BaseModel):
    agent: str = Field(default="VolatilityModelingAgent")
    ticker: str
    iv_rank: float
    volatility_forecast: str
    skew_analysis: str
    term_structure: str


class SentimentReport(BaseModel):
    agent: str = Field(default="SentimentAgent")
    ticker: str
    overall_sentiment_score: float
    key_headlines: List[NewsHeadline]
    kg_derived_insights: Optional[List[dict]] = None


class TechnicalReport(BaseModel):
    agent: str = Field(default="TechnicalAnalyst")
    ticker: str
    classical_summary: str
    key_levels: dict
    dl_oracle_forecast: dict


class FundamentalReport(BaseModel):
    agent: str = Field(default="FundamentalAnalyst")
    ticker: str
    financial_health: dict
    kg_derived_swot: dict


class DebateArgument(BaseModel):
    stance: str
    argument: str


class TradeThesis(BaseModel):
    winning_argument: str
    conviction_level: str
    summary: str
    key_evidence: List[str]


class TradeProposal(BaseModel):
    strategy: str
    direction: str
    expiration: str
    strikes: List[float]
    notes: Optional[str] = None


class RiskAdjustment(BaseModel):
    profile: str
    recommendation: str


class RiskAssessment(BaseModel):
    adjustments: List[RiskAdjustment]
    final_recommendation: str
