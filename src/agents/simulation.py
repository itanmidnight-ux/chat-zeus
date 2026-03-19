"""Agente de simulación ligera por bloques."""
from __future__ import annotations

import gc
from typing import Any

from src.core.scorer import score_solution
from src.utils import clamp


class SimulationAgent:
    def run_scenarios(self, hypotheses: list[dict[str, Any]], iteration: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for block_start in range(0, len(hypotheses), 2):
            block = hypotheses[block_start:block_start + 2]
            for item in block:
                feasibility = clamp(float(item['feasibility']) + 0.02 * iteration, 0.0, 1.0)
                efficiency = clamp(float(item['efficiency']) + 0.01 * iteration, 0.0, 1.0)
                safety = clamp(float(item['safety']) - 0.005 * iteration, 0.0, 1.0)
                results.append({
                    **item,
                    'feasibility': feasibility,
                    'efficiency': efficiency,
                    'safety': safety,
                    'score': score_solution(feasibility, efficiency, safety),
                })
            gc.collect()
        return results
