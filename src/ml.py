"""Orquestador del módulo ML con aprendizaje online, hipótesis y checkpoints."""
from __future__ import annotations

import importlib.metadata
import json
import os
from dataclasses import dataclass
from typing import Any

from src.config import CONFIG
from src.ml_components import CheckpointManager, DataPreprocessor, HypothesisEvaluator, IncrementalLearner, PredictionEngine
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
    variables_considered: list[str]
    uncertainty_drivers: list[str]
    recommendations: list[str]
    reliability_score: float


class LightweightMLModel:
    MODEL_NAME = 'dedicated_online_regressor'
    FEATURE_NAMES = DataPreprocessor.FEATURE_NAMES
    MAX_PREDICTION_OBSERVATIONS = 24
    BACKEND_ENV_VAR = 'CHAT_ZEUS_ML_BACKEND'
    SAFE_BACKEND_LABELS = {
        'heuristic': 'dedicated-online-heuristic',
        'dedicated-online-heuristic': 'dedicated-online-heuristic',
        'tflite_runtime': 'tflite_runtime',
        'tensorflow-lite-compatible': 'tensorflow-lite-compatible',
        'pytorch-mobile-compatible': 'pytorch-mobile-compatible',
    }
    DISTRIBUTION_CANDIDATES = (
        ('tflite-runtime', 'tflite_runtime'),
        ('tflite_runtime', 'tflite_runtime'),
        ('tensorflow', 'tensorflow-lite-compatible'),
        ('torch', 'pytorch-mobile-compatible'),
    )

    def __init__(self, storage: StorageManager):
        self.storage = storage
        self.backend = self._detect_backend()
        self.preprocessor = DataPreprocessor()
        self.learner = IncrementalLearner(storage, self.FEATURE_NAMES)
        self.predictor = PredictionEngine()
        self.hypothesis_evaluator = HypothesisEvaluator()
        self.checkpoints = CheckpointManager(storage, CONFIG.ml_checkpoint_file)
        self.model_state_path = self.checkpoints.path
        self.state = self._load_or_init_state()

    def _detect_backend(self) -> str:
        configured_backend = os.environ.get(self.BACKEND_ENV_VAR, '').strip().lower()
        if configured_backend:
            return self.SAFE_BACKEND_LABELS.get(configured_backend, 'dedicated-online-heuristic')
        if not CONFIG.enable_native_ml_backend_probe:
            return 'dedicated-online-heuristic'
        for distribution_name, label in self.DISTRIBUTION_CANDIDATES:
            try:
                importlib.metadata.distribution(distribution_name)
            except importlib.metadata.PackageNotFoundError:
                continue
            except Exception:
                continue
            return label
        return 'dedicated-online-heuristic'

    def _default_state(self) -> dict[str, Any]:
        return {
            'version': 3,
            'samples_seen': 0,
            'learning_rate': 0.0125,
            'bias': 0.0,
            'weights': {name: 0.0 for name in self.FEATURE_NAMES},
            'feature_mean': {name: 0.0 for name in self.FEATURE_NAMES},
            'feature_scale': {name: 1.0 for name in self.FEATURE_NAMES},
            'loss_ema': 0.0,
            'last_reliability': 0.5,
        }

    def _load_or_init_state(self) -> dict[str, Any]:
        return self.checkpoints.load(self.MODEL_NAME, self._default_state())

    def _persist_state(self) -> None:
        self.checkpoints.save(self.MODEL_NAME, self.state)

    def _feature_stats(self) -> dict[str, dict[str, float]]:
        return {
            feature_name: {
                'mean': float(self.state['feature_mean'].get(feature_name, 0.0)),
                'scale': float(self.state['feature_scale'].get(feature_name, 1.0)),
            }
            for feature_name in self.FEATURE_NAMES
        }

    def _source_weights(self) -> dict[str, float]:
        defaults = {'arxiv': 0.92, 'crossref': 0.88, 'wikipedia': 0.66, 'duckduckgo': 0.52, 'local': 0.9}
        profile = self.storage.source_performance_profile()
        connectivity = self.storage.connectivity_profile()
        merged: dict[str, float] = {}
        for source, default_score in defaults.items():
            usefulness = profile.get(source, default_score)
            success = connectivity.get(source, {}).get('success_rate', 0.8)
            latency = connectivity.get(source, {}).get('avg_latency_ms', 500.0 if source == 'local' else 800.0)
            latency_factor = clamp(1.12 - latency / 4500.0, 0.5, 1.15)
            merged[source] = round(clamp(default_score * 0.45 + usefulness * 0.35 + success * 0.2, 0.25, 0.99) * latency_factor, 4)
        return merged

    def train_from_result(self, simulation_result: dict[str, Any], question: str = '', knowledge_summary: str = '') -> None:
        try:
            preprocessed = self.preprocessor.validate_and_prepare(
                simulation_result,
                question=question,
                knowledge_summary=knowledge_summary,
                source_weights=self._source_weights(),
                stats=self._feature_stats(),
            )
            target = float(simulation_result.get('range_m', 0.0))
            self.storage.append_ml_observation(
                json.dumps(preprocessed.raw_features, ensure_ascii=False),
                target,
                preprocessed.reliability_score,
            )
            self.state = self.learner.update_state(self.state, preprocessed.raw_features, target, preprocessed.reliability_score)
            self._persist_state()
        except MemoryError as exc:
            self.storage.log_error('ml_train', 'memory_error', str(exc), {'backend': self.backend})
        except Exception as exc:
            self.storage.log_error('ml_train', exc.__class__.__name__, str(exc), {'backend': self.backend})

    def predict(self, simulation_result: dict[str, Any], question: str = '', knowledge_summary: str = '') -> HypothesisResult:
        source_weights = self._source_weights()
        try:
            preprocessed = self.preprocessor.validate_and_prepare(
                simulation_result,
                question=question,
                knowledge_summary=knowledge_summary,
                source_weights=source_weights,
                stats=self._feature_stats(),
            )
            observations = self.storage.load_ml_observations(limit=self.MAX_PREDICTION_OBSERVATIONS)
            history_targets = [float(item['target']) for item in observations]
            prediction_data = self.predictor.predict(self.state, preprocessed.raw_features, history_targets, preprocessed.reliability_score)
            hypothesis_data = self.hypothesis_evaluator.evaluate(
                question=question,
                simulation_result=simulation_result,
                prediction=prediction_data['prediction'],
                reliability_score=preprocessed.reliability_score,
                source_weights=source_weights,
                history_targets=history_targets,
                uncertainty_drivers=preprocessed.uncertainty_drivers,
            )
            if observations:
                self.state = self.learner.retrain_in_background(self.state)
                self._persist_state()
            model_state = {
                'samples_seen': int(self.state.get('samples_seen', 0)),
                'loss_ema': round(float(self.state.get('loss_ema', 0.0)), 4),
                'backend': self.backend,
                'weights': {name: round(float(self.state['weights'].get(name, 0.0)), 6) for name in self.FEATURE_NAMES},
            }
            self.storage.log_prediction(
                question=question,
                prediction=prediction_data['prediction'],
                confidence=hypothesis_data['confidence'],
                reliability=preprocessed.reliability_score,
                variables=preprocessed.variables_considered,
                hypotheses=hypothesis_data['hypotheses'],
                recommendations=prediction_data['recommendations'],
            )
            return HypothesisResult(
                prediction=prediction_data['prediction'],
                confidence=hypothesis_data['confidence'],
                hypotheses=hypothesis_data['hypotheses'],
                preferred_domains=hypothesis_data['top_sources'],
                research_intensity=prediction_data['research_intensity'],
                source_weights=source_weights,
                model_state=model_state,
                variables_considered=preprocessed.variables_considered,
                uncertainty_drivers=hypothesis_data['uncertainty_drivers'],
                recommendations=prediction_data['recommendations'],
                reliability_score=preprocessed.reliability_score,
            )
        except MemoryError as exc:
            self.storage.log_error('ml_predict', 'memory_error', str(exc), {'backend': self.backend})
            return self._fallback_result(source_weights, reason='Predicción parcial por límite de memoria.')
        except Exception as exc:
            self.storage.log_error('ml_predict', exc.__class__.__name__, str(exc), {'backend': self.backend})
            return self._fallback_result(source_weights, reason='Predicción parcial por error controlado.')

    def _fallback_result(self, source_weights: dict[str, float], reason: str) -> HypothesisResult:
        return HypothesisResult(
            prediction=0.0,
            confidence=0.22,
            hypotheses=[
                f'{reason} El sistema mantiene la ejecución y conserva checkpoints para continuar más tarde.',
                'La respuesta debe tratarse como preliminar hasta disponer de más datos confiables o una nueva simulación.',
            ],
            preferred_domains=[name for name, _ in sorted(source_weights.items(), key=lambda item: item[1], reverse=True)],
            research_intensity=min(CONFIG.max_external_queries, 12),
            source_weights=source_weights,
            model_state={'samples_seen': int(self.state.get('samples_seen', 0)), 'loss_ema': float(self.state.get('loss_ema', 0.0)), 'backend': self.backend},
            variables_considered=['masa', 'velocidad', 'gravedad', 'arrastre'],
            uncertainty_drivers=['reliability', 'knowledge_support', 'drag'],
            recommendations=['Reintentar con bloques más pequeños o con una simulación simplificada.', 'Validar la hipótesis principal con evidencia adicional antes de usarla como decisión final.'],
            reliability_score=0.2,
        )
