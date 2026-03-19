"""Motor de simulación ligera con checkpoints y tareas pequeñas."""
from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from typing import Any

from src.storage import StorageManager


@dataclass
class SimulationRequest:
    question: str
    payload_mass_kg: float
    fuel_mass_kg: float
    dry_mass_kg: float
    exhaust_velocity_m_s: float
    thrust_n: float
    drag_coefficient: float
    area_m2: float
    air_density_kg_m3: float
    time_step_s: float
    steps: int
    run_id: str | None = None


class SimulationEngine:
    def __init__(self, storage: StorageManager, chunk_size: int = 100):
        self.storage = storage
        self.chunk_size = chunk_size

    def build_request(self, question: str, defaults: dict[str, float | int] | None = None) -> SimulationRequest:
        defaults = defaults or {}
        return SimulationRequest(
            question=question,
            payload_mass_kg=float(defaults.get('payload_mass_kg', 120.0)),
            fuel_mass_kg=float(defaults.get('fuel_mass_kg', 240.0)),
            dry_mass_kg=float(defaults.get('dry_mass_kg', 180.0)),
            exhaust_velocity_m_s=float(defaults.get('exhaust_velocity_m_s', 2800.0)),
            thrust_n=float(defaults.get('thrust_n', 18000.0)),
            drag_coefficient=float(defaults.get('drag_coefficient', 0.45)),
            area_m2=float(defaults.get('area_m2', 1.8)),
            air_density_kg_m3=float(defaults.get('air_density_kg_m3', 1.225)),
            time_step_s=float(defaults.get('time_step_s', 0.2)),
            steps=int(defaults.get('steps', 500)),
            run_id=str(defaults.get('run_id')) if defaults.get('run_id') else None,
        )

    def run(self, request: SimulationRequest, progress_callback=None) -> dict[str, Any]:
        run_id = request.run_id or uuid.uuid4().hex[:12]
        checkpoint = self.storage.load_checkpoint(run_id)
        start_step = int(checkpoint.get('step', 0))
        velocity = float(checkpoint.get('velocity', 0.0))
        altitude = float(checkpoint.get('altitude', 0.0))
        downrange = float(checkpoint.get('downrange', 0.0))
        remaining_fuel = float(checkpoint.get('remaining_fuel', request.fuel_mass_kg))
        max_altitude = float(checkpoint.get('max_altitude', altitude))
        history = checkpoint.get('history', [])

        burn_rate = max(0.01, request.thrust_n / max(request.exhaust_velocity_m_s, 1.0))
        total_initial_mass = request.payload_mass_kg + request.dry_mass_kg + request.fuel_mass_kg
        final_mass = request.payload_mass_kg + request.dry_mass_kg
        delta_v = request.exhaust_velocity_m_s * math.log(max(total_initial_mass / max(final_mass, 1e-6), 1.000001))

        for chunk_start in range(start_step, request.steps, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, request.steps)
            for step in range(chunk_start, chunk_end):
                current_mass = request.payload_mass_kg + request.dry_mass_kg + max(remaining_fuel, 0.0)
                burn = min(remaining_fuel, burn_rate * request.time_step_s)
                remaining_fuel -= burn
                thrust = request.thrust_n if burn > 0 else 0.0
                drag = 0.5 * request.air_density_kg_m3 * request.drag_coefficient * request.area_m2 * velocity * velocity
                drag *= -1 if velocity >= 0 else 1
                gravity = current_mass * 9.81
                net_force = thrust + drag - gravity
                acceleration = net_force / max(current_mass, 1e-6)
                velocity += acceleration * request.time_step_s
                altitude = max(0.0, altitude + velocity * request.time_step_s)
                downrange += max(velocity, 0.0) * request.time_step_s * 0.12
                max_altitude = max(max_altitude, altitude)
                if step % max(1, self.chunk_size // 4) == 0:
                    history.append({'step': step, 'altitude_m': altitude, 'velocity_m_s': velocity})

            payload = {
                'run_id': run_id,
                'step': chunk_end,
                'velocity': velocity,
                'altitude': altitude,
                'downrange': downrange,
                'remaining_fuel': remaining_fuel,
                'max_altitude': max_altitude,
                'history': history[-200:],
            }
            progress = chunk_end / request.steps
            self.storage.save_checkpoint(run_id, payload)
            self.storage.save_run_state(run_id, request.question, 'running', progress, json.dumps(payload))
            if progress_callback:
                progress_callback(run_id, progress)

        burn_time = min(request.fuel_mass_kg / burn_rate, request.steps * request.time_step_s)
        result = {
            'run_id': run_id,
            'delta_v_m_s': round(delta_v, 3),
            'max_altitude_m': round(max_altitude, 3),
            'range_m': round(downrange, 3),
            'burn_time_s': round(burn_time, 3),
            'final_velocity_m_s': round(velocity, 3),
            'remaining_fuel_kg': round(remaining_fuel, 3),
            'payload_mass_kg': request.payload_mass_kg,
            'history': history[-20:],
        }
        self.storage.save_run_state(run_id, request.question, 'completed', 1.0, json.dumps(result))
        self.storage.save_checkpoint(run_id, result)
        return result
