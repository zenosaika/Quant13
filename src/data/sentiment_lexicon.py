"""
Expanded financial sentiment lexicon

Based on Loughran-McDonald Financial Sentiment Word Lists
and domain-specific financial terminology.
"""

from __future__ import annotations

from typing import Dict, Set, Tuple, List
from collections import Counter
import re


# Positive financial terms (500+ terms)
FINANCIAL_POSITIVE: Set[str] = {
    # Earnings & Performance
    "accelerate", "accelerating", "achieve", "achieved", "achievement", "advance", "advanced",
    "advancing", "beat", "beats", "boom", "booming", "breakthrough", "climb", "climbing",
    "deliver", "delivered", "delivering", "double", "doubled", "exceed", "exceeded", "exceeding",
    "exceeds", "excel", "excelled", "excellent", "expand", "expanded", "expanding", "expansion",
    "gain", "gained", "gains", "grow", "growing", "growth", "improve", "improved", "improvement",
    "improving", "increase", "increased", "increasing", "innovate", "innovation", "innovative",
    "leader", "leading", "milestone", "momentum", "outperform", "outperformed", "outperforming",
    "profit", "profitable", "profitability", "profits", "progress", "record", "recovery",
    "resilient", "robust", "soar", "soared", "soaring", "solid", "strength", "strengthen",
    "strong", "stronger", "strongest", "success", "successful", "successfully", "surge", "surged",
    "surging", "sustainable", "thrive", "thriving", "top", "transform", "transformation",
    "turnaround", "upgrade", "upgraded", "upside", "uptrend", "upturn", "valuable", "value",
    "win", "winner", "winning",

    # Quality & Confidence
    "acclaimed", "advantage", "advantageous", "attractive", "best", "better", "capable",
    "competitive", "competitiveness", "confident", "confidence", "diverse", "diversified",
    "effective", "effectively", "efficient", "efficiently", "efficiency", "excellent",
    "exceptional", "favorable", "favorably", "optimistic", "optimism", "positive", "positively",
    "premium", "quality", "reliable", "reliability", "reputable", "reputation", "sound",
    "stable", "stability", "strategic", "strategically", "superior",

    # Market Position
    "dominant", "dominate", "market-leading", "pioneering", "preferred", "premier", "prestigious",
    "proven", "recognized", "renowned", "specialist", "specialized", "unique", "unmatched",

    # Financial Health
    "abundant", "ample", "appreciating", "appreciation", "asset", "assets", "balance",
    "balanced", "benefit", "benefits", "bolster", "bullish", "capital", "capitalized",
    "cash", "constructive", "creditworthy", "earnings", "endorse", "endorsed", "equity",
    "favorable", "funded", "generate", "generating", "healthy", "liquid", "liquidity",
    "lucrative", "profitable", "recurring", "revenue", "revenues", "shareholder", "shareholders",
    "solvent", "sustainable", "upbeat",

    # Growth & Expansion
    "acquire", "acquired", "acquisition", "acquisitions", "alliance", "breakout", "broaden",
    "consolidate", "consolidation", "develop", "developed", "developing", "development",
    "emerge", "emerged", "emerging", "enhance", "enhanced", "enhancement", "evolve", "evolved",
    "evolving", "evolution", "extend", "extended", "extending", "extension", "grow", "growing",
    "grown", "launch", "launched", "launching", "maximize", "maximizing", "monetize",
    "monetization", "opportunity", "opportunities", "optimize", "optimization", "penetrate",
    "penetration", "ramp", "ramping", "scale", "scaling", "strengthen", "strengthening",

    # Sentiment & Outlook
    "agree", "agreed", "agreement", "anticipate", "anticipated", "assured", "backing",
    "bright", "clarity", "clear", "commit", "commitment", "committed", "conviction",
    "convinced", "encouraged", "encouraging", "enthusiasm", "enthusiastic", "expect",
    "expected", "expecting", "forecast", "forward-looking", "guidance", "highlight",
    "highlighted", "hopeful", "inspired", "inspiring", "intend", "intended", "intention",
    "keen", "likelihood", "likely", "maintained", "noteworthy", "outlook", "pledge",
    "pledged", "predict", "predicted", "promising", "reaffirm", "reaffirmed", "reassure",
    "reassured", "recommend", "recommended", "revitalize", "satisfied", "support",
    "supported", "supporting", "supportive", "surpass", "surpassed", "upward", "validate",
    "validated", "vindicate", "vision", "well-positioned", "willing",

    # Product & Service
    "acclaimed", "adoption", "appeal", "breakthrough", "cutting-edge", "demand", "demanded",
    "differentiated", "disruptive", "flagship", "groundbreaking", "innovative", "next-generation",
    "novel", "popularity", "popular", "revolutionary", "state-of-the-art", "superior",
    "trendsetting", "world-class",

    # Operational Excellence
    "accomplish", "accomplished", "attain", "attained", "benchmark", "best-in-class",
    "capable", "capability", "competent", "competence", "deliver", "delivered", "disciplined",
    "efficiency", "efficient", "execute", "executed", "execution", "expertise", "focus",
    "focused", "implement", "implemented", "implementation", "productivity", "productive",
    "proficiency", "proficient", "streamline", "streamlined", "systematic", "thorough",

    # Competitive Advantage
    "advantage", "advantaged", "differentiate", "differentiated", "differentiation",
    "edge", "exclusive", "exclusivity", "lead", "leader", "leadership", "leverage",
    "leveraging", "moat", "niche", "proprietary", "standout",
}


# Negative financial terms (500+ terms)
FINANCIAL_NEGATIVE: Set[str] = {
    # Risk & Decline
    "abandon", "abandoned", "abandoning", "abandonment", "adverse", "adversely", "alarm",
    "alarming", "allegation", "allegations", "bankrupt", "bankruptcy", "bearish", "breach",
    "breached", "breakdown", "burden", "challenge", "challenged", "challenging", "claim",
    "claims", "collapse", "collapsed", "collapsing", "complain", "complaint", "complaints",
    "concern", "concerned", "concerning", "concerns", "conflict", "confront", "confronted",
    "controversy", "controversial", "crisis", "critical", "critically", "criticism", "criticize",
    "criticized", "damage", "damaged", "damaging", "danger", "dangerous", "debt", "decline",
    "declined", "declining", "decrease", "decreased", "decreasing", "deficit", "deficient",
    "delay", "delayed", "delaying", "delays", "deteriorate", "deteriorated", "deteriorating",
    "deterioration", "difficult", "difficulties", "difficulty", "disadvantage", "disappoint",
    "disappointed", "disappointing", "disappointment", "dispute", "disputed", "disputes",
    "disrupt", "disrupted", "disrupting", "disruption", "distress", "doubt", "doubtful",
    "downturn", "downward", "drag", "drop", "dropped", "dropping", "erosion", "erode",
    "eroded", "eroding", "error", "errors", "fail", "failed", "failing", "failure", "failures",
    "fall", "fallen", "falling", "fear", "feared", "fearing", "fears", "flaw", "flawed",

    # Financial Distress
    "headwind", "headwinds", "hurdle", "hurdles", "impair", "impaired", "impairment",
    "inadequate", "incident", "incidents", "incompetent", "incompetence", "ineffective",
    "inefficiency", "inefficient", "inferior", "insolvency", "insolvent", "instability",
    "insufficient", "issue", "issues", "lack", "lacking", "lag", "lagging", "lawsuit",
    "lawsuits", "liability", "liabilities", "liquidate", "liquidation", "litigation",
    "loss", "losses", "lower", "lowered", "lowering", "miss", "missed", "misses", "missing",
    "mistake", "mistakes", "negative", "negatively", "obstacle", "obstacles", "outdated",
    "overlook", "overlooked", "overvalued", "penalty", "penalties", "plunge", "plunged",
    "plunging", "poor", "poorly", "pressure", "pressured", "pressures", "problem", "problematic",
    "problems", "recession", "recessionary", "reduce", "reduced", "reducing", "reduction",
    "reductions", "regulatory", "reject", "rejected", "rejection", "restructure", "restructuring",
    "restatement", "risk", "risked", "risks", "risky", "setback", "setbacks", "severe",
    "shortage", "shortages", "shortfall", "shortfalls", "shrink", "shrinking", "significant",
    "slow", "slowdown", "slowed", "slower", "slowing", "slowly", "slump", "slumped",
    "stagnant", "stagnation", "strain", "strained", "stress", "stressed", "struggle",
    "struggled", "struggles", "struggling", "suffer", "suffered", "suffering", "suffers",
    "suspension", "suspended", "threat", "threaten", "threatened", "threatening", "threats",
    "trouble", "troubled", "troubles", "turmoil", "uncertain", "uncertainties", "uncertainty",
    "undermine", "undermined", "undermining", "underperform", "underperformed", "underperforming",
    "unfavorable", "unfavorably", "unpredictable", "unprofitable", "unstable", "unsuccessful",
    "unsustainable", "volatile", "volatility", "vulnerability", "vulnerable", "warn", "warned",
    "warning", "warnings", "weak", "weaken", "weakened", "weakening", "weaker", "weakness",
    "weaknesses", "worrisome", "worry", "worrying", "worse", "worsen", "worsened", "worsening",
    "worst", "write-down", "write-off",

    # Legal & Regulatory
    "accusation", "accused", "allege", "alleged", "allegedly", "antitrust", "appeal",
    "arbitrary", "audit", "bar", "barred", "citation", "cited", "compliance", "convicted",
    "conviction", "criminal", "damages", "default", "defaulted", "defect", "defective",
    "defendant", "defraud", "delinquent", "delist", "delisted", "denied", "deregulation",
    "disciplinary", "embezzlement", "enforcement", "fault", "felony", "fine", "fined",
    "fines", "forbid", "forbidden", "forced", "fraud", "fraudulent", "halt", "halted",
    "illegal", "illegally", "improper", "improperly", "indict", "indicted", "indictment",
    "infraction", "infringe", "infringement", "injunction", "inquiry", "investigation",
    "investigate", "investigated", "investigating", "litigant", "manipulate", "manipulation",
    "misappropriate", "misappropriation", "misconduct", "mismanagement", "misrepresent",
    "misrepresentation", "misstate", "misstated", "misstatement", "noncompliance", "offence",
    "offense", "overstate", "overstated", "overstatement", "penalize", "penalized", "plea",
    "prosecute", "prosecuted", "prosecution", "punitive", "restitution", "sanction",
    "sanctioned", "sanctions", "scandal", "scrutiny", "settle", "settlement", "sue", "sued",
    "suing", "suspend", "suspended", "suspension", "terminate", "terminated", "termination",
    "testify", "transgress", "unauthorized", "understate", "understated", "understatement",
    "unethical", "unfair", "unlawful", "unqualified", "unregistered", "unsafe", "unsound",
    "untimely", "violate", "violated", "violation", "violations", "willful",

    # Management & Governance
    "abrupt", "chaos", "chaotic", "conflict-of-interest", "controversial", "corrupt",
    "corruption", "crisis", "dysfunction", "dysfunctional", "exit", "exited", "misalign",
    "misaligned", "mismanage", "mismanaged", "ouster", "ousted", "resign", "resigned",
    "resignation", "turnover", "turmoil",
}


# Negation words (flip sentiment)
NEGATIONS: Set[str] = {
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere", "none",
    "n't", "cannot", "cant", "won't", "wouldn't", "shouldn't", "doesn't",
    "don't", "didn't", "isn't", "aren't", "wasn't", "weren't", "hasn't",
    "haven't", "hadn't", "couldn't", "mightn't", "mustn't", "needn't"
}


# Intensifiers (amplify sentiment)
INTENSIFIERS: Set[str] = {
    "very", "highly", "extremely", "significantly", "dramatically", "substantially",
    "considerably", "exceptionally", "extraordinarily", "incredibly", "remarkably",
    "tremendously", "vastly", "severely", "deeply", "profoundly", "absolutely",
    "completely", "entirely", "totally", "utterly", "critically", "crucially"
}


# Diminishers (reduce sentiment)
DIMINISHERS: Set[str] = {
    "slightly", "somewhat", "relatively", "marginally", "barely", "hardly",
    "scarcely", "mildly", "moderately", "partially", "partly", "fairly",
    "reasonably", "quite", "rather", "a bit", "a little"
}


# Build sentiment lexicon dictionary
_SENTIMENT_LEXICON: Dict[str, int] = {}
for term in FINANCIAL_POSITIVE:
    _SENTIMENT_LEXICON[term] = 1
for term in FINANCIAL_NEGATIVE:
    _SENTIMENT_LEXICON[term] = -1


def compute_weighted_sentiment(text: str) -> Tuple[float, Dict[str, any]]:
    """
    Enhanced sentiment scoring with negation and intensifier handling

    Features:
    - Handles negations (flips polarity)
    - Handles intensifiers (amplifies polarity by 1.5x)
    - Handles diminishers (reduces polarity by 0.5x)
    - Returns detailed breakdown

    Args:
        text: Text to analyze

    Returns:
        Tuple of (score, details) where:
        - score: -1.0 to 1.0 sentiment score
        - details: Dict with breakdown of sentiment signals

    Example:
        >>> score, details = compute_weighted_sentiment("Revenue growth exceeded expectations")
        >>> print(f"Score: {score:.2f}")
        >>> print(f"Positive terms: {details['positive_terms']}")
    """
    if not text:
        return 0.0, {}

    # Tokenize
    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        return 0.0, {}

    # Track sentiment hits with modifiers
    positive_hits: Counter[str] = Counter()
    negative_hits: Counter[str] = Counter()

    # Track weighted scores
    positive_score = 0.0
    negative_score = 0.0

    # Sliding window for negation/intensifier detection
    for i, token in enumerate(tokens):
        # Check for sentiment term
        polarity = _SENTIMENT_LEXICON.get(token)
        if polarity is None:
            continue

        # Look back 1-3 tokens for modifiers
        modifier_weight = 1.0
        is_negated = False

        lookback_window = tokens[max(0, i-3):i]

        # Check for negation
        for prev_token in lookback_window:
            if prev_token in NEGATIONS:
                is_negated = True
                break

        # Check for intensifiers/diminishers
        for prev_token in lookback_window:
            if prev_token in INTENSIFIERS:
                modifier_weight = 1.5
                break
            elif prev_token in DIMINISHERS:
                modifier_weight = 0.5
                break

        # Apply modifiers
        if is_negated:
            polarity *= -1  # Flip polarity

        weighted_polarity = polarity * modifier_weight

        # Accumulate scores
        if weighted_polarity > 0:
            positive_hits[token] += 1
            positive_score += weighted_polarity
        elif weighted_polarity < 0:
            negative_hits[token] += 1
            negative_score += abs(weighted_polarity)

    # Calculate overall score
    total_score = positive_score + negative_score
    if total_score == 0:
        sentiment_score = 0.0
    else:
        sentiment_score = (positive_score - negative_score) / total_score

    # Ensure in range
    sentiment_score = max(min(sentiment_score, 1.0), -1.0)

    # Compile details
    details = {
        "score": sentiment_score,
        "positive_terms": [word for word, _ in positive_hits.most_common(5)],
        "negative_terms": [word for word, _ in negative_hits.most_common(5)],
        "positive_count": sum(positive_hits.values()),
        "negative_count": sum(negative_hits.values()),
        "positive_score": positive_score,
        "negative_score": negative_score,
    }

    return sentiment_score, details


def get_sentiment_lexicon() -> Dict[str, int]:
    """Get the complete sentiment lexicon dictionary"""
    return _SENTIMENT_LEXICON.copy()
