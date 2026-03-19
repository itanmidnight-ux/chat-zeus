"""Planificador de investigación por prioridad y criticidad."""
from __future__ import annotations

from typing import Any

CRITICAL_TOKENS = {
    'seguridad': 1.0,
    'riesgo': 0.95,
    'energía': 0.9,
    'energia': 0.9,
    'coste': 0.75,
    'estructura': 0.8,
    'propulsión': 0.85,
    'propulsion': 0.85,
}


def build_research_plan(question: str, tasks: list[str]) -> list[dict[str, Any]]:
    lowered = question.lower()
    plan: list[dict[str, Any]] = []
    for item in tasks:
        priority = CRITICAL_TOKENS.get(item.lower(), 0.6)
        if item.lower() in lowered:
            priority += 0.15
        plan.append({
            'task': item,
            'priority': round(min(priority, 1.0), 2),
            'critical': priority >= 0.85,
        })
    return sorted(plan, key=lambda row: (row['priority'], row['critical']), reverse=True)
