"""Módulo ML ligero sin dependencias pesadas: regresión incremental y heurísticas."""
from __future__ import annotations

import json
from dataclasses import dataclass
from statistics import mean
from typing import Any

from src.storage import StorageManager


@dataclass
class HypothesisResult:
    prediction: float
    confidence: float
    hypotheses: list[str]


class LightweightMLModel:
    def __init__(self, storage: StorageManager):
        self.storage = storage

    def _feature_vector(self, simulation_result: dict[str, Any]) -> dict[str, float]:
        return {
            'delta_v': float(simulation_result.get('delta_v_m_s', 0.0)),
            'max_altitude': float(simulation_result.get('max_altitude_m', 0.0)),
            'burn_time': float(simulation_result.get('burn_time_s', 0.0)),
            'payload_mass': float(simulation_result.get('payload_mass_kg', 0.0)),
        }

    def train_from_result(self, simulation_result: dict[str, Any]) -> None:
        features = self._feature_vector(simulation_result)
        target = float(simulation_result.get('range_m', 0.0))
        self.storage.append_ml_observation(json.dumps(features), target)

    def predict(self, simulation_result: dict[str, Any]) -> HypothesisResult:
        current = self._feature_vector(simulation_result)
        observations = self.storage.load_ml_observations()
        if not observations:
            baseline = current['delta_v'] * max(current['burn_time'], 1.0)
            return HypothesisResult(
                prediction=baseline,
                confidence=0.25,
                hypotheses=[
                    'No hay historial suficiente; se usa una heurística basada en delta-v y tiempo de combustión.',
                    'Más datos históricos mejorarán la precisión de la predicción.',
                ],
            )

        targets = [float(item['target']) for item in observations]
        avg_target = mean(targets)
        feature_strength = (current['delta_v'] + current['max_altitude'] * 0.01 + current['payload_mass']) / 3.0
        prediction = 0.6 * avg_target + 0.4 * feature_strength
        confidence = min(0.9, 0.35 + len(observations) * 0.05)
        hypotheses = [
            'La tendencia histórica indica que aumentar el delta-v mejora el alcance estimado.',
            'Reducir masa estructural probablemente eleve la altitud máxima en próximas iteraciones.',
            'Conviene contrastar esta hipótesis con una corrida de optimización adicional.',
        ]
        return HypothesisResult(prediction=prediction, confidence=confidence, hypotheses=hypotheses)
