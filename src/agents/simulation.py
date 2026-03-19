"""Simulation agent: bounded lightweight numeric approximations."""
from __future__ import annotations

import gc
from typing import Any

from src.core.scorer import score_solution
from src.utils import clamp


class SimulationAgent:
    def run(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        try:
            feasibility = clamp(float(hypothesis.get('feasibility', 0.6)), 0.0, 1.0)
            efficiency = clamp(float(hypothesis.get('efficiency', 0.6)), 0.0, 1.0)
            safety = clamp(float(hypothesis.get('safety', 0.7)), 0.0, 1.0)
            for _ in range(3):
                feasibility = clamp(feasibility * 0.995 + 0.01, 0.0, 1.0)
                efficiency = clamp(efficiency * 0.99 + 0.012, 0.0, 1.0)
                safety = clamp(safety * 0.997 + 0.006, 0.0, 1.0)
            return {
                'feasibility': round(feasibility, 4),
                'efficiency': round(efficiency, 4),
                'safety': round(safety, 4),
                'score': score_solution({'feasibility': feasibility, 'efficiency': efficiency, 'safety': safety}),
            }
        except Exception:
            return {'feasibility': 0.5, 'efficiency': 0.5, 'safety': 0.6, 'score': 0.53}

    def run_scenarios(self, hypotheses: list[dict[str, Any]], iteration: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for block_start in range(0, len(hypotheses), 2):
            for item in hypotheses[block_start:block_start + 2]:
                scenario = self.run({
                    **item,
                    'feasibility': clamp(float(item.get('feasibility', 0.6)) + 0.02 * iteration, 0.0, 1.0),
                    'efficiency': clamp(float(item.get('efficiency', 0.6)) + 0.01 * iteration, 0.0, 1.0),
                    'safety': clamp(float(item.get('safety', 0.7)) - 0.005 * iteration, 0.0, 1.0),
                })
                results.append({**item, **scenario})
            gc.collect()
        return results
