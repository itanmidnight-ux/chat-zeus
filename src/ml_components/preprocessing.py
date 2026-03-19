"""Preprocesamiento, validación y filtros de confiabilidad para el módulo ML."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.utils import clamp, sanitize_text


@dataclass
class PreprocessedBatch:
    raw_features: dict[str, float]
    normalized_features: dict[str, float]
    encoded_features: dict[str, float]
    reliability_score: float
    discarded_fields: list[str]
    warnings: list[str]
    variables_considered: list[str]
    uncertainty_drivers: list[str]


class DataPreprocessor:
    FEATURE_NAMES = [
        'delta_v', 'max_altitude', 'burn_time', 'payload_mass', 'chem_efficiency',
        'gravity', 'drag', 'fuel_reactivity', 'math_complexity', 'knowledge_support',
    ]
    CATEGORY_ENCODINGS = {
        'physics': 0.25,
        'chemistry': 0.5,
        'mathematics': 0.75,
        'systems': 1.0,
    }

    def validate_and_prepare(
        self,
        simulation_result: dict[str, Any],
        *,
        question: str = '',
        knowledge_summary: str = '',
        source_weights: dict[str, float] | None = None,
        stats: dict[str, dict[str, float]] | None = None,
    ) -> PreprocessedBatch:
        source_weights = source_weights or {}
        stats = stats or {}
        chemistry = simulation_result.get('chemistry', {})
        math_data = simulation_result.get('math', {})
        raw_features = {
            'delta_v': self._safe_number(simulation_result.get('delta_v_m_s'), 'delta_v_m_s'),
            'max_altitude': self._safe_number(simulation_result.get('max_altitude_m'), 'max_altitude_m'),
            'burn_time': self._safe_number(simulation_result.get('burn_time_s'), 'burn_time_s'),
            'payload_mass': self._safe_number(simulation_result.get('payload_mass_kg'), 'payload_mass_kg'),
            'chem_efficiency': self._safe_number(chemistry.get('estimated_efficiency'), 'estimated_efficiency'),
            'gravity': self._safe_number(simulation_result.get('gravity_m_s2', 9.81), 'gravity_m_s2', default=9.81),
            'drag': self._safe_number(simulation_result.get('drag_newton', simulation_result.get('drag_coefficient', 0.0)), 'drag_newton', default=0.0),
            'fuel_reactivity': self._safe_number(chemistry.get('fuel_reactivity', chemistry.get('mixture_ratio', 0.0)), 'fuel_reactivity', default=0.0),
            'math_complexity': self._safe_number(math_data.get('complexity_score', self._infer_math_complexity(question)), 'math_complexity', default=0.0),
            'knowledge_support': self._safe_number(len(knowledge_summary) / 256.0, 'knowledge_support', default=0.0),
        }
        discarded_fields: list[str] = []
        warnings: list[str] = []
        for key, value in list(raw_features.items()):
            if math.isnan(value) or math.isinf(value):
                raw_features[key] = 0.0
                discarded_fields.append(key)
                warnings.append(f'Se descartó {key} por contener un valor no finito.')
            elif abs(value) > 1e9:
                raw_features[key] = clamp(value, -1e9, 1e9)
                warnings.append(f'Se limitó {key} por exceder el rango seguro.')

        encoded_features = dict(raw_features)
        for category, weight in self.CATEGORY_ENCODINGS.items():
            encoded_features[f'category_{category}'] = weight if category in sanitize_text(question.lower()) else 0.0

        normalized_features: dict[str, float] = {}
        for feature_name, value in raw_features.items():
            feature_stats = stats.get(feature_name, {})
            mean_value = float(feature_stats.get('mean', 0.0))
            scale_value = max(1.0, float(feature_stats.get('scale', 1.0)))
            normalized_features[feature_name] = (value - mean_value) / scale_value

        reliability_score = self._compute_reliability(simulation_result, source_weights, discarded_fields)
        variables_considered = self._variables_considered(simulation_result, chemistry, math_data)
        uncertainty_drivers = self._uncertainty_drivers(raw_features, reliability_score)
        return PreprocessedBatch(
            raw_features=raw_features,
            normalized_features=normalized_features,
            encoded_features=encoded_features,
            reliability_score=reliability_score,
            discarded_fields=discarded_fields,
            warnings=warnings,
            variables_considered=variables_considered,
            uncertainty_drivers=uncertainty_drivers,
        )

    def _safe_number(self, value: Any, field_name: str, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _infer_math_complexity(self, question: str) -> float:
        lowered = question.lower()
        score = 0.0
        for keyword, delta in {
            'integral': 0.8,
            'derivada': 0.6,
            'matriz': 0.7,
            'tensor': 0.9,
            'ecuación': 0.5,
            'optimización': 0.75,
        }.items():
            if keyword in lowered:
                score += delta
        return round(min(score, 3.0), 4)

    def _compute_reliability(
        self,
        simulation_result: dict[str, Any],
        source_weights: dict[str, float],
        discarded_fields: list[str],
    ) -> float:
        local_weight = 0.72
        external_weight = sum(source_weights.values()) / max(len(source_weights), 1) if source_weights else 0.55
        uncertainty = float(simulation_result.get('uncertainty_index', 0.25) or 0.25)
        checkpoint_bonus = 0.05 if simulation_result.get('resource_profile', {}).get('resumed_from_checkpoint') else 0.0
        conflict_penalty = min(0.25, len(discarded_fields) * 0.04)
        score = local_weight * 0.45 + external_weight * 0.35 + (1.0 - uncertainty) * 0.2 + checkpoint_bonus - conflict_penalty
        return round(clamp(score, 0.05, 0.99), 4)

    def _variables_considered(
        self,
        simulation_result: dict[str, Any],
        chemistry: dict[str, Any],
        math_data: dict[str, Any],
    ) -> list[str]:
        variables = ['masa', 'velocidad', 'gravedad', 'arrastre', 'combustibles', 'eficiencia química']
        if chemistry:
            variables.extend(['reacciones', 'mezcla', 'presión efectiva'])
        if math_data or simulation_result.get('mode') == 'general_analysis':
            variables.extend(['integrales', 'derivadas', 'matrices'])
        return variables

    def _uncertainty_drivers(self, features: dict[str, float], reliability_score: float) -> list[str]:
        drivers: list[tuple[str, float]] = [
            ('payload_mass', abs(features.get('payload_mass', 0.0)) * 0.0015),
            ('drag', abs(features.get('drag', 0.0)) * 0.002),
            ('fuel_reactivity', 1.0 - min(abs(features.get('fuel_reactivity', 0.0)) / 5.0, 1.0)),
            ('knowledge_support', 1.0 - min(features.get('knowledge_support', 0.0) / 10.0, 1.0)),
            ('reliability', 1.0 - reliability_score),
        ]
        drivers.sort(key=lambda item: item[1], reverse=True)
        return [name for name, _ in drivers[:3]]
