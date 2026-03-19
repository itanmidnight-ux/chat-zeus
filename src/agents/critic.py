"""Critic agent: evaluates realism and assigns a normalized score."""
from __future__ import annotations

from typing import Any

from src.core.scorer import score_solution
from src.utils import clamp


class CriticAgent:
    def evaluate_hypothesis(self, hypothesis: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
        try:
            feasibility = clamp(float(results.get('feasibility', hypothesis.get('feasibility', 0.0))), 0.0, 1.0)
            efficiency = clamp(float(results.get('efficiency', hypothesis.get('efficiency', 0.0))), 0.0, 1.0)
            safety = clamp(float(results.get('safety', hypothesis.get('safety', 0.0))), 0.0, 1.0)
            penalty = 0.0
            if safety < 0.6:
                penalty += 0.08
            if feasibility < 0.55:
                penalty += 0.05
            if efficiency < 0.45:
                penalty += 0.03
            adjusted_safety = clamp(safety - penalty, 0.0, 1.0)
            return {'risk_penalty': round(penalty, 3), 'final_score': score_solution({'feasibility': feasibility, 'efficiency': efficiency, 'safety': adjusted_safety})}
        except Exception:
            return {'risk_penalty': 0.2, 'final_score': 0.0}

    def evaluate(self, scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reviewed = []
        for item in scenarios:
            reviewed.append({**item, **self.evaluate_hypothesis(item, item)})
        return sorted(reviewed, key=lambda row: row['final_score'], reverse=True)
