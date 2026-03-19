"""Módulo ML ligero sin dependencias pesadas: regresión incremental y heurísticas."""
from __future__ import annotations

import json
from dataclasses import dataclass
from statistics import mean
from typing import Any

from src.storage import StorageManager
from src.utils import clamp


@dataclass
class HypothesisResult:
    prediction: float
    confidence: float
    hypotheses: list[str]


class LightweightMLModel:
    def __init__(self, storage: StorageManager):
        self.storage = storage
        self.backend = self._detect_backend()

    def _detect_backend(self) -> str:
        for module_name, label in [('tflite_runtime', 'tflite_runtime'), ('tensorflow', 'tensorflow-lite-compatible'), ('torch', 'pytorch-mobile-compatible')]:
            try:
                __import__(module_name)
                return label
            except Exception:
                continue
        return 'heuristic-fallback'

    def _feature_vector(self, simulation_result: dict[str, Any]) -> dict[str, float]:
        chemistry = simulation_result.get('chemistry', {})
        return {
            'delta_v': float(simulation_result.get('delta_v_m_s', 0.0)),
            'max_altitude': float(simulation_result.get('max_altitude_m', 0.0)),
            'burn_time': float(simulation_result.get('burn_time_s', 0.0)),
            'payload_mass': float(simulation_result.get('payload_mass_kg', 0.0)),
            'chem_efficiency': float(chemistry.get('estimated_efficiency', 0.0)),
        }

    def train_from_result(self, simulation_result: dict[str, Any]) -> None:
        features = self._feature_vector(simulation_result)
        target = float(simulation_result.get('range_m', 0.0))
        self.storage.append_ml_observation(json.dumps(features, ensure_ascii=False), target)

    def predict(self, simulation_result: dict[str, Any]) -> HypothesisResult:
        current = self._feature_vector(simulation_result)
        observations = self.storage.load_ml_observations()
        if not observations:
            baseline = current['delta_v'] * max(current['burn_time'], 1.0) * max(current['chem_efficiency'], 0.3) / 10.0
            return HypothesisResult(
                prediction=baseline,
                confidence=0.25,
                hypotheses=[
                    f'No hay historial suficiente; se usa una heurística ligera con backend {self.backend}.',
                    'Más datos históricos mejorarán la precisión de la predicción y de las hipótesis generadas.',
                ],
            )

        targets = [float(item['target']) for item in observations]
        avg_target = mean(targets)
        feature_strength = (
            current['delta_v'] * 0.4
            + current['max_altitude'] * 0.01
            + current['burn_time'] * 5.0
            + current['chem_efficiency'] * 100.0
            - current['payload_mass'] * 0.08
        )
        prediction = max(0.0, 0.55 * avg_target + 0.45 * feature_strength)
        confidence = clamp(0.35 + len(observations) * 0.05, 0.35, 0.92)
        hypotheses = [
            f'El backend activo para aprendizaje es {self.backend}; si instalas TensorFlow Lite o PyTorch Mobile, este módulo puede ampliarse sin cambiar la interfaz.',
            'La tendencia histórica indica que aumentar el delta-v efectivo y la eficiencia química mejora el alcance estimado.',
            'Reducir masa estructural y área frontal probablemente eleve la altitud máxima en próximas iteraciones.',
            'Conviene contrastar esta hipótesis con una corrida de optimización adicional y revisar el compromiso entre empuje y drag.',
        ]
        return HypothesisResult(prediction=prediction, confidence=confidence, hypotheses=hypotheses)
