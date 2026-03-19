"""Research planner with explicit priority ordering."""
from __future__ import annotations

from typing import Any

CRITICAL_TOKENS = {
    'seguridad': 1.0,
    'safety': 1.0,
    'riesgo': 0.95,
    'risk': 0.95,
    'energía': 0.9,
    'energia': 0.9,
    'energy': 0.9,
    'coste': 0.75,
    'cost': 0.75,
    'estructura': 0.8,
    'structure': 0.8,
    'propulsión': 0.85,
    'propulsion': 0.85,
}


def prioritize_tasks(subtasks: list[str], question: str = '') -> list[dict[str, Any]]:
    try:
        lowered = question.lower()
        plan: list[dict[str, Any]] = []
        for item in subtasks:
            priority = CRITICAL_TOKENS.get(item.lower(), 0.6)
            if item.lower() in lowered:
                priority += 0.15
            plan.append({'task': item, 'priority': round(min(priority, 1.0), 2), 'critical': priority >= 0.85})
        return sorted(plan, key=lambda row: (row['priority'], row['critical']), reverse=True)
    except Exception:
        return [{'task': item, 'priority': 0.5, 'critical': False} for item in subtasks]


def build_research_plan(question: str, tasks: list[str]) -> list[dict[str, Any]]:
    return prioritize_tasks(tasks, question=question)
