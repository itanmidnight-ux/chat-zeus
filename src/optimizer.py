"""Optimización iterativa simple mediante muestreo aleatorio guiado."""
from __future__ import annotations

import random
from typing import Any

from src.simulation import SimulationEngine


class IterativeOptimizer:
    def __init__(self, simulation_engine: SimulationEngine, seed: int = 7):
        self.simulation_engine = simulation_engine
        self.random = random.Random(seed)

    def optimize(self, question: str, iterations: int = 6, progress_callback=None) -> dict[str, Any]:
        best_result: dict[str, Any] | None = None
        best_params: dict[str, float] | None = None
        best_score = float('-inf')
        for idx in range(iterations):
            params = {
                'payload_mass_kg': self.random.uniform(80, 180),
                'fuel_mass_kg': self.random.uniform(180, 360),
                'dry_mass_kg': self.random.uniform(130, 220),
                'exhaust_velocity_m_s': self.random.uniform(2400, 3200),
                'thrust_n': self.random.uniform(12000, 26000),
                'drag_coefficient': self.random.uniform(0.22, 0.6),
                'area_m2': self.random.uniform(0.9, 2.5),
                'mixture_ratio': self.random.uniform(2.1, 3.0),
                'chamber_temperature_k': self.random.uniform(2800, 3600),
                'steps': 320,
            }
            request = self.simulation_engine.build_request(question, params)
            result = self.simulation_engine.run(request)
            score = result['max_altitude_m'] + result['range_m'] * 0.2 + result['delta_v_m_s'] * 0.8
            if score > best_score:
                best_score = score
                best_result = result
                best_params = params
            if progress_callback:
                progress_callback('optimizer', (idx + 1) / iterations)
        return {
            'best_parameters': best_params or {},
            'best_result': best_result or {},
            'iterations': iterations,
            'objective': '0.8 * delta_v + max_altitude + 0.2 * range',
            'best_score': round(best_score, 3) if best_score != float('-inf') else 0.0,
        }
