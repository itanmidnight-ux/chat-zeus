"""Agente generador de hipótesis."""
from __future__ import annotations

from typing import Any


class ReasoningAgent:
    def generate_hypotheses(self, question: str, tasks: list[str], research: dict[str, Any], intent: str) -> list[dict[str, Any]]:
        base = research.get('summary', '')
        hypotheses = []
        for index, task in enumerate(tasks[:4], start=1):
            hypotheses.append({
                'id': f'h{index}',
                'task': task,
                'proposal': f'Priorizar {task} con un diseño simple, seguro y escalable.',
                'feasibility': max(0.45, 0.78 - index * 0.07),
                'efficiency': max(0.4, 0.74 - index * 0.05),
                'safety': max(0.5, 0.82 - index * 0.04),
                'context': base[:240],
                'intent': intent,
            })
        if not hypotheses:
            hypotheses.append({
                'id': 'h1', 'task': 'solución', 'proposal': f'Resolver "{question}" con una estrategia gradual y validable.',
                'feasibility': 0.65, 'efficiency': 0.62, 'safety': 0.72, 'context': base[:240], 'intent': intent,
            })
        return hypotheses
