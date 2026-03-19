"""Agente crítico para penalizar riesgos."""
from __future__ import annotations

from typing import Any

from src.core.scorer import score_solution
from src.utils import clamp


class CriticAgent:
    def evaluate(self, scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reviewed = []
        for item in scenarios:
            risk_penalty = 0.0
            if item['safety'] < 0.6:
                risk_penalty += 0.08
            if item['feasibility'] < 0.55:
                risk_penalty += 0.05
            final_score = score_solution(item['feasibility'], item['efficiency'], clamp(item['safety'] - risk_penalty, 0.0, 1.0))
            reviewed.append({
                **item,
                'risk_penalty': round(risk_penalty, 3),
                'final_score': final_score,
            })
        return sorted(reviewed, key=lambda row: row['final_score'], reverse=True)
