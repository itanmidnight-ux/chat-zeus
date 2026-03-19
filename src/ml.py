"""Módulo ML dedicado al programa con aprendizaje online y priorización adaptativa."""
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
    preferred_domains: list[str]
    research_intensity: int
    source_weights: dict[str, float]
    model_state: dict[str, Any]


class LightweightMLModel:
    MODEL_NAME = 'dedicated_online_regressor'
    FEATURE_NAMES = ['delta_v', 'max_altitude', 'burn_time', 'payload_mass', 'chem_efficiency']

    def __init__(self, storage: StorageManager):
        self.storage = storage
        self.backend = self._detect_backend()
        self.state = self._load_or_init_state()

    def _detect_backend(self) -> str:
        for module_name, label in [('tflite_runtime', 'tflite_runtime'), ('tensorflow', 'tensorflow-lite-compatible'), ('torch', 'pytorch-mobile-compatible')]:
            try:
                __import__(module_name)
                return label
            except Exception:
                continue
        return 'dedicated-online-heuristic'

    def _load_or_init_state(self) -> dict[str, Any]:
        state = self.storage.load_model_state(self.MODEL_NAME)
        if state:
            return state
        state = {
            'version': 1,
            'samples_seen': 0,
            'learning_rate': 0.015,
            'bias': 0.0,
            'weights': {name: 0.0 for name in self.FEATURE_NAMES},
            'feature_mean': {name: 0.0 for name in self.FEATURE_NAMES},
            'feature_scale': {name: 1.0 for name in self.FEATURE_NAMES},
            'loss_ema': 0.0,
        }
        self.storage.save_model_state(self.MODEL_NAME, state)
        return state

    def _feature_vector(self, simulation_result: dict[str, Any]) -> dict[str, float]:
        chemistry = simulation_result.get('chemistry', {})
        return {
            'delta_v': float(simulation_result.get('delta_v_m_s', 0.0)),
            'max_altitude': float(simulation_result.get('max_altitude_m', 0.0)),
            'burn_time': float(simulation_result.get('burn_time_s', 0.0)),
            'payload_mass': float(simulation_result.get('payload_mass_kg', 0.0)),
            'chem_efficiency': float(chemistry.get('estimated_efficiency', 0.0)),
        }

    def _normalized_features(self, features: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for name in self.FEATURE_NAMES:
            mean_value = float(self.state['feature_mean'].get(name, 0.0))
            scale_value = max(float(self.state['feature_scale'].get(name, 1.0)), 1.0)
            normalized[name] = (features.get(name, 0.0) - mean_value) / scale_value
        return normalized

    def _predict_linear(self, normalized_features: dict[str, float]) -> float:
        score = float(self.state.get('bias', 0.0))
        for name in self.FEATURE_NAMES:
            score += float(self.state['weights'].get(name, 0.0)) * normalized_features.get(name, 0.0)
        return score

    def train_from_result(self, simulation_result: dict[str, Any]) -> None:
        features = self._feature_vector(simulation_result)
        target = float(simulation_result.get('range_m', 0.0))
        self.storage.append_ml_observation(json.dumps(features, ensure_ascii=False), target)
        self._online_update(features, target)

    def _online_update(self, features: dict[str, float], target: float) -> None:
        samples_seen = int(self.state.get('samples_seen', 0)) + 1
        learning_rate = float(self.state.get('learning_rate', 0.015))
        for name in self.FEATURE_NAMES:
            old_mean = float(self.state['feature_mean'].get(name, 0.0))
            delta = features[name] - old_mean
            new_mean = old_mean + delta / samples_seen
            self.state['feature_mean'][name] = new_mean
            self.state['feature_scale'][name] = max(1.0, float(self.state['feature_scale'].get(name, 1.0)) * 0.99 + abs(delta) * 0.01)
        normalized = self._normalized_features(features)
        prediction = self._predict_linear(normalized)
        error = target - prediction
        self.state['bias'] = float(self.state.get('bias', 0.0)) + learning_rate * error * 0.05
        for name in self.FEATURE_NAMES:
            self.state['weights'][name] = float(self.state['weights'].get(name, 0.0)) + learning_rate * error * normalized[name] * 0.01
        previous_loss = float(self.state.get('loss_ema', 0.0))
        self.state['loss_ema'] = previous_loss * 0.92 + abs(error) * 0.08
        self.state['samples_seen'] = samples_seen
        self.storage.save_model_state(self.MODEL_NAME, self.state)

    def _source_weights(self) -> dict[str, float]:
        defaults = {'arxiv': 0.92, 'crossref': 0.88, 'wikipedia': 0.66, 'duckduckgo': 0.52}
        profile = self.storage.source_performance_profile()
        connectivity = self.storage.connectivity_profile()
        merged: dict[str, float] = {}
        for source, default_score in defaults.items():
            usefulness = profile.get(source, default_score)
            success = connectivity.get(source, {}).get('success_rate', 0.75)
            latency = connectivity.get(source, {}).get('avg_latency_ms', 800.0)
            latency_factor = clamp(1.12 - latency / 4500.0, 0.5, 1.15)
            merged[source] = round(clamp(default_score * 0.45 + usefulness * 0.35 + success * 0.2, 0.25, 0.99) * latency_factor, 4)
        return merged

    def predict(self, simulation_result: dict[str, Any]) -> HypothesisResult:
        current = self._feature_vector(simulation_result)
        observations = self.storage.load_ml_observations()
        source_weights = self._source_weights()
        normalized = self._normalized_features(current)
        linear_prediction = self._predict_linear(normalized)
        heuristic_prediction = (
            current['delta_v'] * 0.22
            + current['max_altitude'] * 0.04
            + current['burn_time'] * 2.5
            + current['chem_efficiency'] * 110.0
            - current['payload_mass'] * 0.05
        )
        if not observations:
            prediction = max(0.0, heuristic_prediction)
            return HypothesisResult(
                prediction=prediction,
                confidence=0.28,
                preferred_domains=[name for name, _ in sorted(source_weights.items(), key=lambda item: item[1], reverse=True)],
                research_intensity=22,
                source_weights=source_weights,
                model_state={'samples_seen': 0, 'loss_ema': 0.0, 'backend': self.backend},
                hypotheses=[
                    f'No hay historial suficiente; se activa un modelo dedicado inicial con backend {self.backend}.',
                    'A partir de cada simulación, el modelo guarda pesos propios y se reentrena online solo con datos del programa.',
                    'La investigación debe usar redundancia de fuentes mientras el modelo sigue madurando.',
                ],
            )

        targets = [float(item['target']) for item in observations]
        avg_target = mean(targets)
        blended_prediction = max(0.0, 0.35 * avg_target + 0.35 * heuristic_prediction + 0.30 * linear_prediction)
        samples_seen = int(self.state.get('samples_seen', 0))
        loss_ema = float(self.state.get('loss_ema', 0.0))
        confidence = clamp(0.32 + min(samples_seen, 40) * 0.012 - min(loss_ema / 5000.0, 0.18), 0.32, 0.94)
        research_intensity = int(clamp(18 + (1.0 - confidence) * 18 + min(samples_seen, 20) * 0.6, 18, 48))
        preferred_domains = [name for name, _ in sorted(source_weights.items(), key=lambda item: item[1], reverse=True)]
        hypotheses = [
            f'El backend activo para aprendizaje es {self.backend}; el modelo principal es un regresor online dedicado al dominio del programa.',
            'Cada simulación actualiza pesos, medias y escalas internas para que el ML aprenda específicamente de este sistema y no de datos genéricos.',
            'El historial de conectividad y la utilidad observada de cada fuente afectan directamente la priorización de búsquedas externas.',
            'La salida del modelo sigue siendo una ayuda de priorización avanzada; aún requiere verificación científica externa antes de tomar decisiones reales.',
        ]
        return HypothesisResult(
            prediction=blended_prediction,
            confidence=confidence,
            hypotheses=hypotheses,
            preferred_domains=preferred_domains,
            research_intensity=research_intensity,
            source_weights=source_weights,
            model_state={
                'samples_seen': samples_seen,
                'loss_ema': round(loss_ema, 4),
                'backend': self.backend,
                'weights': {name: round(float(self.state['weights'].get(name, 0.0)), 6) for name in self.FEATURE_NAMES},
            },
        )
