"""Reasoning agent: generates hypotheses only."""
from __future__ import annotations

from typing import Any


class ReasoningAgent:
    def reason(self, research_data: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            keywords = research_data.get('keywords', [])
            topic = research_data.get('topic', 'problema complejo')
            hypotheses: list[dict[str, Any]] = []
            base_tasks = keywords[:3] or ['objetivo', 'restricciones', 'validación']
            for index, task in enumerate(base_tasks, start=1):
                hypotheses.append({
                    'id': f'h{index}',
                    'task': task,
                    'proposal': f'Organizar {topic} alrededor de {task} con etapas simples y verificables.',
                    'feasibility': max(0.45, 0.8 - index * 0.07),
                    'efficiency': max(0.4, 0.76 - index * 0.05),
                    'safety': max(0.5, 0.83 - index * 0.04),
                })
            return hypotheses or [{'id': 'h1', 'task': 'solución', 'proposal': f'Resolver {topic} de forma gradual.', 'feasibility': 0.64, 'efficiency': 0.61, 'safety': 0.73}]
        except Exception:
            return [{'id': 'h1', 'task': 'solución', 'proposal': 'Resolver el problema de forma gradual.', 'feasibility': 0.6, 'efficiency': 0.55, 'safety': 0.7}]

    def generate_hypotheses(self, question: str, tasks: list[str], research: dict[str, Any], intent: str) -> list[dict[str, Any]]:
        payload = {'topic': question, 'keywords': tasks[:4], 'research': research, 'intent': intent}
        return self.reason(payload)
