"""Experiment loop engine for bounded evaluation of hypotheses."""
from __future__ import annotations

from typing import Any

from src.agents.critic import CriticAgent
from src.agents.simulation import SimulationAgent


def run_experiments(hypotheses: list[dict[str, Any]], max_iterations: int = 5) -> dict[str, Any]:
    simulator = SimulationAgent()
    critic = CriticAgent()
    best: dict[str, Any] | None = None
    try:
        bounded = hypotheses[: max(1, min(10, max_iterations))]
        for item in bounded:
            scenario = simulator.run(hypothesis=item)
            evaluated = critic.evaluate_hypothesis(item, scenario)
            if best is None or evaluated.get('final_score', 0.0) > best.get('final_score', 0.0):
                best = {**item, **scenario, **evaluated}
        return best or {'proposal': 'No se encontró una solución suficiente.', 'final_score': 0.0, 'task': 'general'}
    except Exception:
        return {'proposal': 'No se encontró una solución suficiente.', 'final_score': 0.0, 'task': 'general'}
