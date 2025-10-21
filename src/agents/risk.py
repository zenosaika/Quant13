from __future__ import annotations

from typing import List

from src.models.schemas import RiskAdjustment, RiskAssessment, TradeProposal, TradeThesis, VolatilityReport


class RiskManagementTeam:
    def __init__(self) -> None:
        pass

    def assess(self, trade: TradeProposal, thesis: TradeThesis, volatility: VolatilityReport) -> RiskAssessment:
        adjustments: List[RiskAdjustment] = []

        iv_rank = volatility.iv_rank
        conviction = thesis.conviction_level.lower()

        if iv_rank > 70:
            adjustments.append(RiskAdjustment(profile="Safe", recommendation="Reduce position sizing due to elevated implied volatility."))
        else:
            adjustments.append(RiskAdjustment(profile="Safe", recommendation="Position size within standard risk parameters."))

        adjustments.append(RiskAdjustment(profile="Neutral", recommendation=f"Ensure Greeks align with {conviction} conviction level."))

        if conviction in {"high", "elevated"}:
            adjustments.append(RiskAdjustment(profile="Risky", recommendation="Consider scaling into position if liquidity permits."))
        else:
            adjustments.append(RiskAdjustment(profile="Risky", recommendation="Maintain base exposure without leverage increase."))

        final_note = "Proceed with trade after aligning with risk overlays." if iv_rank < 80 else "Proceed cautiously; consider hedging due to high IV rank."
        return RiskAssessment(adjustments=adjustments, final_recommendation=final_note)
