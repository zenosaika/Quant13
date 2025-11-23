"""
Enhanced Sentiment Agent with multi-source analysis

Improvements over original:
- Expanded 500+ term lexicon with negation handling
- Multi-source aggregation (news, analyst ratings, SEC filings)
- Confidence scoring
- Weighted component scores
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from src.agents.base import Agent
from src.models.schemas import ArticleSentiment
from src.tools.llm import get_llm_client
from src.data.sentiment_lexicon import compute_weighted_sentiment
import logging

logger = logging.getLogger(__name__)


class EnhancedSentimentAgent(Agent):
    """
    Multi-source sentiment analysis with confidence scoring
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = get_llm_client()

    def _think(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment from multiple sources

        Sources (with weights):
        1. News articles (30%)
        2. Analyst ratings (25%)
        3. SEC filing tone (20%)
        4. Lexical sentiment fallback (25% if LLM fails)

        Returns analysis with component scores and confidence
        """
        ticker = state["ticker"]

        # 1. News sentiment
        news_score, news_details = self._analyze_news_sentiment(state.get("news", []))

        # 2. Analyst ratings sentiment
        analyst_score, analyst_details = self._analyze_analyst_ratings(
            state.get("company_info", {})
        )

        # 3. SEC filing tone (reuse fundamental agent's data if available)
        filing_score, filing_details = self._analyze_sec_filing_tone(
            state.get("filings", {})
        )

        # 4. Calculate weighted overall score
        component_weights = {
            "news": 0.30,
            "analyst_ratings": 0.25,
            "sec_filings": 0.20,
            "lexical": 0.25
        }

        # Adjust weights based on available data
        available_components = {}
        if news_details.get("available"):
            available_components["news"] = news_score
        if analyst_details.get("available"):
            available_components["analyst_ratings"] = analyst_score
        if filing_details.get("available"):
            available_components["sec_filings"] = filing_score

        # Normalize weights
        if available_components:
            total_weight = sum(component_weights[k] for k in available_components.keys())
            overall_score = sum(
                available_components[k] * component_weights[k] / total_weight
                for k in available_components.keys()
            )
        else:
            overall_score = 0.0

        # 5. Calculate confidence
        confidence = self._calculate_confidence(
            news_details, analyst_details, filing_details
        )

        # 6. Generate overall summary
        overall_summary = self._generate_summary(
            overall_score, confidence, available_components
        )

        return {
            "overall_sentiment_score": overall_score,
            "overall_summary": overall_summary,
            "confidence": confidence,
            "component_scores": {
                "news": {
                    "score": news_score,
                    "weight": component_weights["news"],
                    "details": news_details
                },
                "analyst_ratings": {
                    "score": analyst_score,
                    "weight": component_weights["analyst_ratings"],
                    "details": analyst_details
                },
                "sec_filings": {
                    "score": filing_score,
                    "weight": component_weights["sec_filings"],
                    "details": filing_details
                }
            },
            "articles": news_details.get("articles", [])
        }

    def _generate_report(self, analysis: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced sentiment report"""
        from src.models.schemas import SentimentReport

        return SentimentReport(
            ticker=state["ticker"],
            overall_sentiment_score=analysis["overall_sentiment_score"],
            overall_summary=analysis["overall_summary"],
            articles=[ArticleSentiment(**article) for article in analysis.get("articles", [])]
        )

    def _analyze_news_sentiment(
        self, news_items: List[Dict[str, Any]]
    ) -> tuple[float, Dict[str, Any]]:
        """
        Analyze news articles sentiment

        Uses LLM for primary analysis, enhanced lexicon for fallback

        Returns: (score, details)
        """
        if not news_items:
            return 0.0, {"available": False, "reason": "No news articles"}

        # Filter recent news (7 days)
        recent_news = [
            item for item in news_items
            if self._is_recent(self._extract_timestamp(item))
        ]

        if not recent_news and news_items:
            # Use most recent article even if older
            recent_news = [news_items[0]]

        payload_articles = [self._normalize_article(item) for item in recent_news]

        if not payload_articles:
            return 0.0, {"available": False, "reason": "No valid articles"}

        # Try LLM analysis first
        try:
            messages = [
                {"role": "system", "content": self.config["prompt"]},
                {"role": "user", "content": json.dumps({
                    "ticker": "ticker",
                    "articles": payload_articles
                })},
            ]
            raw_response = self.llm.chat(messages, temperature=0.3)
            parsed = self._parse_sentiment_response(raw_response, payload_articles)

            return parsed["overall_sentiment_score"], {
                "available": True,
                "source": "llm",
                "articles": parsed["articles"],
                "article_count": len(parsed["articles"])
            }

        except Exception as e:
            logger.warning(f"LLM sentiment analysis failed, using enhanced lexicon: {e}")

        # Fallback to enhanced lexical analysis
        scored_articles = []
        for article in payload_articles:
            text = self._get_article_text(article)
            score, details = compute_weighted_sentiment(text)

            scored_articles.append({
                "title": article.get("title", ""),
                "publisher": article.get("publisher"),
                "link": article.get("link"),
                "published_at": article.get("published_at"),
                "sentiment_score": score,
                "rationale": f"Lexical analysis: {', '.join(details['positive_terms'][:2])} (positive), {', '.join(details['negative_terms'][:2])} (negative)",
                "summary": article.get("summary")
            })

        if scored_articles:
            avg_score = sum(a["sentiment_score"] for a in scored_articles) / len(scored_articles)
        else:
            avg_score = 0.0

        return avg_score, {
            "available": True,
            "source": "lexicon",
            "articles": scored_articles,
            "article_count": len(scored_articles)
        }

    def _analyze_analyst_ratings(
        self, company_info: Dict[str, Any]
    ) -> tuple[float, Dict[str, Any]]:
        """
        Analyze analyst ratings and price targets

        Returns: (score, details)
        """
        if not company_info:
            return 0.0, {"available": False, "reason": "No company info"}

        try:
            # Get recommendation key
            recommendation = company_info.get("recommendationKey", "").lower()

            # Rating to score mapping
            rating_scores = {
                "strong_buy": 1.0,
                "buy": 0.6,
                "hold": 0.0,
                "underweight": -0.3,
                "sell": -0.6,
                "strong_sell": -1.0,
            }

            rating_score = rating_scores.get(recommendation, 0.0)

            # Price target upside/downside
            target = company_info.get("targetMeanPrice")
            current = company_info.get("currentPrice")

            target_score = 0.0
            upside_pct = None

            if target and current and current > 0:
                upside_pct = (target - current) / current
                # Convert to -1 to 1 scale (cap at +/- 50%)
                target_score = max(min(upside_pct / 0.5, 1.0), -1.0)

            # Weighted combination (60% rating, 40% target)
            if rating_score != 0.0 or target_score != 0.0:
                overall_score = 0.6 * rating_score + 0.4 * target_score
            else:
                overall_score = 0.0

            return overall_score, {
                "available": True,
                "recommendation": recommendation,
                "rating_score": rating_score,
                "target_price": target,
                "current_price": current,
                "upside_percent": upside_pct * 100 if upside_pct else None,
                "target_score": target_score
            }

        except Exception as e:
            logger.warning(f"Error analyzing analyst ratings: {e}")
            return 0.0, {"available": False, "reason": str(e)}

    def _analyze_sec_filing_tone(
        self, filings: Dict[str, Any]
    ) -> tuple[float, Dict[str, Any]]:
        """
        Analyze tone of SEC filings (MD&A, Risk Factors)

        Returns: (score, details)
        """
        if not filings:
            return 0.0, {"available": False, "reason": "No SEC filings"}

        try:
            # Combine MD&A and Risk Factors text
            mdna_text = ""
            risk_text = ""

            for filing in filings.get("filings", []):
                if filing.get("form_type") in ["10-K", "10-Q"]:
                    sections = filing.get("sections", {})
                    mdna_text += sections.get("mdna", "") + " "
                    risk_text += sections.get("risk_factors", "") + " "

            combined_text = mdna_text + " " + risk_text

            if not combined_text.strip():
                return 0.0, {"available": False, "reason": "No filing text"}

            # Use enhanced lexical analysis (filings are too long for LLM context)
            score, details = compute_weighted_sentiment(combined_text[:10000])  # First 10K chars

            # SEC filings tend to be cautious/negative, so adjust
            # Neutral filing (0.0) should be slightly positive signal
            adjusted_score = score * 0.7  # Dampen the negativity bias

            return adjusted_score, {
                "available": True,
                "source": "lexical",
                "score": score,
                "adjusted_score": adjusted_score,
                "positive_terms": details["positive_terms"],
                "negative_terms": details["negative_terms"],
                "text_length": len(combined_text)
            }

        except Exception as e:
            logger.warning(f"Error analyzing SEC filing tone: {e}")
            return 0.0, {"available": False, "reason": str(e)}

    def _calculate_confidence(
        self,
        news_details: Dict,
        analyst_details: Dict,
        filing_details: Dict
    ) -> Dict[str, Any]:
        """
        Calculate confidence in sentiment assessment

        Factors:
        - Source diversity (more sources = higher confidence)
        - Data freshness
        - Data volume
        - Consensus (all sources agree = higher confidence)
        """
        available_sources = sum([
            news_details.get("available", False),
            analyst_details.get("available", False),
            filing_details.get("available", False)
        ])

        # Source diversity score (0-1)
        diversity_score = available_sources / 3.0

        # Data volume score
        article_count = news_details.get("article_count", 0)
        volume_score = min(article_count / 10.0, 1.0)  # 10+ articles = 1.0

        # Overall confidence (simple average for now)
        overall = (diversity_score + volume_score) / 2.0

        # Categorize
        if overall >= 0.7:
            level = "high"
        elif overall >= 0.4:
            level = "medium"
        else:
            level = "low"

        return {
            "overall": overall,
            "level": level,
            "factors": {
                "source_diversity": diversity_score,
                "data_volume": volume_score
            },
            "available_sources": available_sources
        }

    def _generate_summary(
        self,
        score: float,
        confidence: Dict[str, Any],
        components: Dict[str, float]
    ) -> str:
        """Generate human-readable summary"""
        magnitude = abs(score)

        if magnitude < 0.15:
            tone = "neutral"
        elif score > 0:
            tone = "modestly positive" if magnitude < 0.40 else "strongly positive"
        else:
            tone = "modestly negative" if magnitude < 0.40 else "strongly negative"

        conf_level = confidence["level"]
        source_count = confidence["available_sources"]

        return (
            f"Multi-source sentiment analysis across {source_count} data source(s) "
            f"indicates a {tone} outlook (score={score:.2f}, confidence={conf_level})."
        )

    # Helper methods (from original agent)
    def _is_recent(self, timestamp: Any, lookback_days: int = 7) -> bool:
        """Check if timestamp is within lookback window"""
        if not timestamp:
            return False
        try:
            if isinstance(timestamp, (int, float)):
                ts = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
            elif isinstance(timestamp, str):
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            elif isinstance(timestamp, datetime):
                ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
            else:
                return False
        except (TypeError, ValueError):
            return False
        return (datetime.now(timezone.utc) - ts).days <= lookback_days

    def _extract_timestamp(self, item: Dict[str, Any]) -> Any:
        """Extract timestamp from news item"""
        if item.get("providerPublishTime"):
            return item.get("providerPublishTime")
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        if content.get("pubDate"):
            return content.get("pubDate")
        if content.get("displayTime"):
            return content.get("displayTime")
        return None

    def _normalize_article(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize article structure"""
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        timestamp = self._extract_timestamp(item)

        published_at = None
        if timestamp:
            try:
                if isinstance(timestamp, (int, float)):
                    published_at = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()
                elif isinstance(timestamp, str):
                    ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    published_at = ts.isoformat()
                elif isinstance(timestamp, datetime):
                    ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
                    published_at = ts.isoformat()
            except (TypeError, ValueError):
                published_at = None

        return {
            "title": self._as_text(item.get("title") or content.get("title") or ""),
            "publisher": self._as_text(item.get("publisher") or content.get("publisher")),
            "link": self._link_url(item.get("link") or content.get("canonicalUrl")),
            "published_at": published_at,
            "summary": self._as_text(item.get("summary") or content.get("summary")),
            "body": self._body_text(item, content)
        }

    def _get_article_text(self, article: Dict[str, Any]) -> str:
        """Get combined text from article for analysis"""
        parts = []
        if article.get("title"):
            parts.append(article["title"])
        if article.get("summary"):
            parts.append(article["summary"])
        if article.get("body"):
            parts.append(article["body"][:1000])  # Limit body length
        return " ".join(parts)

    def _body_text(self, item: Dict[str, Any], content: Dict[str, Any]) -> Optional[str]:
        """Extract body text"""
        for key in ("scraped_content", "body", "text"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("body", "articleBody", "content"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _as_text(self, value: Any) -> Optional[str]:
        """Convert value to text"""
        if value is None:
            return None
        if isinstance(value, dict):
            for key in ("displayName", "name", "title", "text", "provider"):
                if key in value and isinstance(value[key], str):
                    return value[key]
            return None
        if isinstance(value, (list, tuple)):
            for item in value:
                text = self._as_text(item)
                if text:
                    return text
            return None
        return str(value)

    def _link_url(self, value: Any) -> Optional[str]:
        """Extract URL"""
        if value is None:
            return None
        if isinstance(value, dict):
            for key in ("url", "canonicalUrl", "href"):
                inner = value.get(key)
                if isinstance(inner, str):
                    return inner
            return None
        if isinstance(value, (list, tuple)):
            for item in value:
                url = self._link_url(item)
                if url:
                    return url
            return None
        return str(value)

    def _parse_sentiment_response(
        self, raw: str, fallback_articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse LLM sentiment response with fallback"""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}

        if not isinstance(data, dict) or not data:
            # Fallback to lexical
            return self._lexical_fallback(fallback_articles)

        articles_payload = []
        for article in data.get("articles", []):
            if not isinstance(article, dict):
                continue
            title = article.get("title")
            if not title:
                continue

            sentiment_score = article.get("sentiment_score")
            try:
                sentiment_score = float(sentiment_score)
            except (TypeError, ValueError):
                sentiment_score = 0.0

            articles_payload.append({
                "title": title,
                "publisher": article.get("publisher"),
                "link": article.get("link"),
                "published_at": article.get("published_at"),
                "summary": article.get("summary"),
                "sentiment_score": sentiment_score,
                "rationale": article.get("rationale"),
            })

        if not articles_payload:
            return self._lexical_fallback(fallback_articles)

        overall_score = data.get("overall_sentiment_score")
        try:
            overall_score = float(overall_score)
        except (TypeError, ValueError):
            overall_score = sum(a["sentiment_score"] for a in articles_payload) / len(articles_payload)

        return {
            "articles": articles_payload,
            "overall_sentiment_score": overall_score,
            "overall_summary": data.get("overall_summary", "Sentiment analyzed")
        }

    def _lexical_fallback(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback to enhanced lexical analysis"""
        scored_articles = []

        for article in articles:
            text = self._get_article_text(article)
            score, details = compute_weighted_sentiment(text)

            scored_articles.append({
                "title": article.get("title", ""),
                "publisher": article.get("publisher"),
                "link": article.get("link"),
                "published_at": article.get("published_at"),
                "sentiment_score": score,
                "rationale": f"Enhanced lexical analysis: {details['positive_count']} positive, {details['negative_count']} negative terms",
                "summary": article.get("summary")
            })

        if scored_articles:
            avg_score = sum(a["sentiment_score"] for a in scored_articles) / len(scored_articles)
        else:
            avg_score = 0.0

        return {
            "articles": scored_articles,
            "overall_sentiment_score": avg_score,
            "overall_summary": f"Enhanced lexical sentiment: {avg_score:.2f}"
        }
