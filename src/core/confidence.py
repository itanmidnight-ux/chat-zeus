"""Confidence calibration helpers for multi-step answer generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ConfidenceReport:
    score: float
    band: str
    factors: dict[str, float]
    recommended_behavior: str


class ConfidenceEngine:
    def evaluate(
        self,
        *,
        intent_scores: dict[str, float],
        selected_intent: str,
        verification_score: float,
        memory_hit: bool,
        source_count: int,
        route_confidence: float,
        failure_penalty: float,
    ) -> ConfidenceReport:
        intent_conf = float(intent_scores.get(selected_intent, 0.2))
        score = (
            intent_conf * 0.35
            + verification_score * 0.3
            + min(1.0, source_count / 3.0) * 0.15
            + route_confidence * 0.15
            + (0.08 if memory_hit else 0.0)
            - failure_penalty * 0.2
        )
        score = round(max(0.05, min(0.99, score)), 4)
        if score >= 0.8:
            band = 'high'
            behavior = 'answer_directly'
        elif score >= 0.55:
            band = 'medium'
            behavior = 'answer_with_caution'
        elif score >= 0.3:
            band = 'low'
            behavior = 'answer_with_uncertainty'
        else:
            band = 'very_low'
            behavior = 'fallback_or_ask_for_precision'
        return ConfidenceReport(
            score=score,
            band=band,
            factors={
                'intent': round(intent_conf, 4),
                'verification': round(verification_score, 4),
                'sources': round(min(1.0, source_count / 3.0), 4),
                'route': round(route_confidence, 4),
                'failure_penalty': round(failure_penalty, 4),
            },
            recommended_behavior=behavior,
        )
