from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base import Agent
from src.models.schemas import ArticleSentiment, SentimentReport
from src.tools.llm import get_llm_client


class SentimentAgent(Agent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = get_llm_client()

    def _think(self, state: Dict[str, Any]) -> Dict[str, Any]:
        news_items: List[Dict[str, Any]] = state.get("news", [])
        recent_news = [item for item in news_items if _is_recent(_extract_timestamp(item))]
        payload_articles = [_normalize_article(item) for item in recent_news]
        if not payload_articles and news_items:
            payload_articles = [_normalize_article(news_items[0])]

        messages = [
            {"role": "system", "content": self.config["prompt"]},
            {"role": "user", "content": json.dumps({"ticker": state["ticker"], "articles": payload_articles})},
        ]
        raw_response = self.llm.chat(messages, temperature=0.3)
        parsed = _parse_sentiment_response(raw_response, payload_articles)

        return {
            "articles": parsed["articles"],
            "overall_sentiment_score": parsed.get("overall_sentiment_score", 0.0),
            "overall_summary": parsed.get("overall_summary", "No substantive sentiment identified."),
            "raw": raw_response,
        }

    def _generate_report(self, analysis: Dict[str, Any], state: Dict[str, Any]) -> SentimentReport:
        return SentimentReport(
            ticker=state["ticker"],
            overall_sentiment_score=analysis["overall_sentiment_score"],
            overall_summary=analysis["overall_summary"],
            articles=[ArticleSentiment(**article) for article in analysis["articles"]],
        )


def _is_recent(timestamp: Any, lookback_days: int = 7) -> bool:
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


def _normalize_article(item: Dict[str, Any]) -> Dict[str, Any]:
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    timestamp = _extract_timestamp(item)
    published_at = None
    body = _body_text(item, content)
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
    raw_summary = _as_text(item.get("summary") or content.get("summary"))
    summary = _short_summary(raw_summary, body)

    return {
        "title": _as_text(item.get("title") or item.get("headline") or content.get("title") or ""),
        "publisher": _as_text(item.get("publisher") or content.get("publisher") or content.get("provider")),
        "link": _link_url(item.get("link") or content.get("canonicalUrl") or content.get("link")),
        "published_at": published_at,
        "summary": summary,
        "body": body,
    }


def _body_text(item: Dict[str, Any], content: Dict[str, Any]) -> Optional[str]:
    for key in ("scraped_content", "body", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("body", "articleBody", "content"):
        value = content.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _short_summary(summary: Optional[str], body: Optional[str], max_sentences: int = 2, max_chars: int = 280) -> Optional[str]:
    text = (summary or "").strip()
    if not text:
        text = (body or "").strip()
    if not text:
        return summary

    sentences = _split_sentences(text)
    if not sentences:
        truncated = text[:max_chars].strip()
        return truncated or summary

    parts: List[str] = []
    for sentence in sentences:
        cleaned = re.sub(r"\s+", " ", sentence).strip()
        if not cleaned:
            continue
        parts.append(cleaned)
        joined = " ".join(parts)
        if len(parts) >= max_sentences or len(joined) >= max_chars:
            break

    excerpt = re.sub(r"\s+", " ", " ".join(parts)).strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rstrip()
    return excerpt


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    return re.split(r"(?<=[.!?])\s+", text)


def _local_sentiment_from_articles(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored_articles: List[Dict[str, Any]] = []
    for article in articles:
        short_summary = _short_summary(article.get("summary"), article.get("body"))
        text = (article.get("body") or short_summary or article.get("title") or "").strip()
        if not text:
            scored_articles.append({
                "title": article.get("title", ""),
                "publisher": article.get("publisher"),
                "link": article.get("link"),
                "published_at": article.get("published_at"),
                "sentiment_score": 0.0,
                "rationale": "Insufficient textual content available for scoring.",
                "summary": short_summary,
            })
            continue

        score, positives, negatives = _lexical_sentiment(text)
        rationale_bits = []
        if positives:
            rationale_bits.append(f"Positive cues: {', '.join(list(positives)[:3])}")
        if negatives:
            rationale_bits.append(f"Negative cues: {', '.join(list(negatives)[:3])}")
        if not rationale_bits:
            rationale_bits.append("Neutral language detected.")

        scored_articles.append({
            "title": article.get("title", ""),
            "publisher": article.get("publisher"),
            "link": article.get("link"),
            "published_at": article.get("published_at"),
            "sentiment_score": score,
            "rationale": " ".join(rationale_bits),
            "summary": short_summary,
        })

    if not scored_articles:
        return {}

    overall = sum(item["sentiment_score"] for item in scored_articles) / len(scored_articles)
    summary = _summarize_overall(overall, len(scored_articles))

    return {
        "articles": scored_articles,
        "overall_sentiment_score": overall,
        "overall_summary": summary,
    }


def _lexical_sentiment(text: str) -> Tuple[float, List[str], List[str]]:
    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        return 0.0, [], []

    positive_hits: Counter[str] = Counter()
    negative_hits: Counter[str] = Counter()
    for token in tokens:
        polarity = _SENTIMENT_LEXICON.get(token)
        if polarity is None:
            continue
        if polarity > 0:
            positive_hits[token] += 1
        elif polarity < 0:
            negative_hits[token] += 1

    pos_total = sum(positive_hits.values())
    neg_total = sum(negative_hits.values())
    if pos_total + neg_total == 0:
        return 0.0, [], []

    score = (pos_total - neg_total) / (pos_total + neg_total)
    score = max(min(score, 1.0), -1.0)
    return score, [word for word, _ in positive_hits.most_common()], [word for word, _ in negative_hits.most_common()]


def _summarize_overall(score: float, count: int) -> str:
    magnitude = abs(score)
    if magnitude < 0.15:
        tone = "neutral"
    elif score > 0:
        tone = "modestly positive" if magnitude < 0.35 else "strongly positive"
    else:
        tone = "modestly negative" if magnitude < 0.35 else "strongly negative"
    return f"Local sentiment analysis across {count} articles indicates a {tone} tone (score={score:.2f})."


_POSITIVE_TERMS = {
    "accelerate", "advances", "advancing", "beat", "beats", "booming", "bullish", "climb", "confidence",
    "double", "exceed", "expansion", "growth", "improve", "improved", "improving", "innovation", "lead",
    "opportunity", "outperform", "profit", "profits", "record", "resilient", "robust", "soar", "strength",
    "strong", "surge", "upbeat", "wins",
}

_NEGATIVE_TERMS = {
    "abandon", "bearish", "collapse", "concern", "concerns", "contraction", "decline", "declines", "drop",
    "fall", "fear", "headwind", "headwinds", "loss", "losses", "lower", "miss", "misses", "plunge", "risk",
    "risks", "slowdown", "slump", "struggle", "struggles", "uncertain", "warning",
}

_SENTIMENT_LEXICON: Dict[str, int] = {term: 1 for term in _POSITIVE_TERMS}
_SENTIMENT_LEXICON.update({term: -1 for term in _NEGATIVE_TERMS})


def _extract_timestamp(item: Dict[str, Any]) -> Any:
    if item.get("providerPublishTime"):
        return item.get("providerPublishTime")
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    if content.get("pubDate"):
        return content.get("pubDate")
    if content.get("displayTime"):
        return content.get("displayTime")
    return None


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("displayName", "name", "title", "text", "provider", "id"):
            if key in value and isinstance(value[key], str):
                return value[key]
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _as_text(item)
            if text:
                return text
        return None
    return str(value)


def _link_url(value: Any) -> Optional[str]:
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
            url = _link_url(item)
            if url:
                return url
        return None
    return str(value)


def _parse_sentiment_response(raw: str, fallback_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    if not isinstance(data, dict):
        data = {}

    if not data:
        local = _local_sentiment_from_articles(fallback_articles)
        if local:
            return local

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
        local = _local_sentiment_from_articles(fallback_articles)
        if local:
            return local
        for article in fallback_articles:
            articles_payload.append({
                "title": article.get("title", ""),
                "publisher": article.get("publisher"),
                "link": article.get("link"),
                "published_at": article.get("published_at"),
                "summary": article.get("summary"),
                "sentiment_score": 0.0,
                "rationale": "Insufficient data; defaulting to neutral sentiment.",
            })

    overall_score = data.get("overall_sentiment_score")
    try:
        overall_score = float(overall_score)
    except (TypeError, ValueError):
        overall_score = sum(a["sentiment_score"] for a in articles_payload) / len(articles_payload) if articles_payload else 0.0

    overall_summary = data.get("overall_summary") or "Sentiment evaluated as neutral given limited signals."

    return {
        "articles": articles_payload,
        "overall_sentiment_score": overall_score,
        "overall_summary": overall_summary,
    }
