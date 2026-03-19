"""Optimización iterativa simple mediante muestreo aleatorio guiado."""
from __future__ import annotations

import random
from typing import Any

from src.simulation import SimulationEngine


class IterativeOptimizer:
    def __init__(self, simulation_engine: SimulationEngine, seed: int = 7):
        self.simulation_engine = simulation_engine
        self.random = random.Random(seed)

    def optimize(self, question: str, iterations: int = 4, progress_callback=None) -> dict[str, Any]:
        best_result: dict[str, Any] | None = None
        best_params: dict[str, float] | None = None
        for idx in range(iterations):
            params = {
                'payload_mass_kg': self.random.uniform(80, 180),
                'fuel_mass_kg': self.random.uniform(180, 360),
                'dry_mass_kg': self.random.uniform(140, 220),
                'exhaust_velocity_m_s': self.random.uniform(2400, 3200),
                'thrust_n': self.random.uniform(12000, 24000),
                'drag_coefficient': self.random.uniform(0.25, 0.6),
                'area_m2': self.random.uniform(1.0, 2.5),
                'steps': 300,
            }
            request = self.simulation_engine.build_request(question, params)
            result = self.simulation_engine.run(request)
            score = result['max_altitude_m'] + result['range_m'] * 0.2
            if not best_result or score > (best_result['max_altitude_m'] + best_result['range_m'] * 0.2):
                best_result = result
                best_params = params
            if progress_callback:
                progress_callback('optimizer', (idx + 1) / iterations)
        return {
            'best_parameters': best_params or {},
            'best_result': best_result or {},
            'iterations': iterations,
        }
