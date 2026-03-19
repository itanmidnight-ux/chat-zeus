"""Optimización iterativa simple mediante muestreo aleatorio guiado y checkpoints."""
from __future__ import annotations

import json
import random
import uuid
from typing import Any

from src.config import CONFIG
from src.simulation import SimulationEngine
from src.storage import StorageManager


class IterativeOptimizer:
    def __init__(self, simulation_engine: SimulationEngine, storage: StorageManager, seed: int = 7):
        self.simulation_engine = simulation_engine
        self.storage = storage
        self.random = random.Random(seed)

    def optimize(self, question: str, iterations: int | None = None, progress_callback=None) -> dict[str, Any]:
        effective_iterations = max(4, int(iterations or CONFIG.optimizer_iterations))
        run_id = f"opt_{uuid.uuid5(uuid.NAMESPACE_DNS, question).hex[:12]}"
        checkpoint = self.storage.load_checkpoint(run_id)
        start_idx = int(checkpoint.get('iteration', 0))
        best_result: dict[str, Any] | None = checkpoint.get('best_result')
        best_params: dict[str, float] | None = checkpoint.get('best_parameters')
        best_score = float(checkpoint.get('best_score', float('-inf')))

        for idx in range(start_idx, effective_iterations):
            params = {
                'payload_mass_kg': self.random.uniform(80, 180),
                'fuel_mass_kg': self.random.uniform(180, 420),
                'dry_mass_kg': self.random.uniform(130, 240),
                'exhaust_velocity_m_s': self.random.uniform(2400, 3400),
                'thrust_n': self.random.uniform(12000, 30000),
                'drag_coefficient': self.random.uniform(0.22, 0.6),
                'area_m2': self.random.uniform(0.9, 2.5),
                'mixture_ratio': self.random.uniform(2.1, 3.0),
                'chamber_temperature_k': self.random.uniform(2800, 3600),
                'steps': min(CONFIG.hard_step_cap, max(CONFIG.default_steps, 640)),
            }
            request = self.simulation_engine.build_request(question, params)
            result = self.simulation_engine.run(request)
            score = result['max_altitude_m'] + result['range_m'] * 0.2 + result['delta_v_m_s'] * 0.8
            if score > best_score:
                best_score = score
                best_result = result
                best_params = params

            payload = {
                'run_id': run_id,
                'iteration': idx + 1,
                'best_parameters': best_params or {},
                'best_result': best_result or {},
                'iterations': effective_iterations,
                'objective': '0.8 * delta_v + max_altitude + 0.2 * range',
                'best_score': round(best_score, 3) if best_score != float('-inf') else 0.0,
            }
            self.storage.save_checkpoint(run_id, payload)
            self.storage.save_run_state(run_id, question, 'running', (idx + 1) / effective_iterations, json.dumps(payload, ensure_ascii=False))
            if progress_callback:
                progress_callback(run_id, (idx + 1) / effective_iterations)

        result = {
            'run_id': run_id,
            'best_parameters': best_params or {},
            'best_result': best_result or {},
            'iterations': effective_iterations,
            'objective': '0.8 * delta_v + max_altitude + 0.2 * range',
            'best_score': round(best_score, 3) if best_score != float('-inf') else 0.0,
        }
        self.storage.save_run_state(run_id, question, 'completed', 1.0, json.dumps(result, ensure_ascii=False))
        self.storage.save_checkpoint(run_id, result)
        return result
