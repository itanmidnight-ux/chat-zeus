"""Generación de predicciones y recomendaciones para el chatbot."""
from __future__ import annotations

from statistics import mean
from typing import Any

from src.config import CONFIG
from src.utils import clamp


class PredictionEngine:
    def predict(self, state: dict[str, Any], features: dict[str, float], history_targets: list[float], reliability: float) -> dict[str, Any]:
        normalized = {}
        linear = float(state.get('bias', 0.0))
        for name, value in features.items():
            mean_value = float(state['feature_mean'].get(name, 0.0))
            scale_value = max(float(state['feature_scale'].get(name, 1.0)), 1.0)
            normalized[name] = (value - mean_value) / scale_value
            linear += float(state['weights'].get(name, 0.0)) * normalized[name]
        heuristic = (
            features.get('delta_v', 0.0) * 0.18
            + features.get('max_altitude', 0.0) * 0.035
            + features.get('burn_time', 0.0) * 2.2
            + features.get('chem_efficiency', 0.0) * 120.0
            - features.get('payload_mass', 0.0) * 0.045
            - features.get('drag', 0.0) * 0.03
        )
        historical = mean(history_targets) if history_targets else heuristic
        prediction = max(0.0, heuristic * 0.34 + linear * 0.33 + historical * 0.33)
        research_intensity = int(clamp(10 + (1.0 - reliability) * 12 + min(len(history_targets), 20) * 0.4, 8, CONFIG.max_external_queries))
        recommendations = [
            'Priorizar nuevas simulaciones en bloques pequeños y reanudar desde checkpoints cuando la consulta sea larga.',
            'Reforzar las variables con mayor incertidumbre antes de aceptar la hipótesis principal como definitiva.',
        ]
        if features.get('drag', 0.0) > 0.7:
            recommendations.append('Reducir arrastre o revisar coeficiente aerodinámico para disminuir la incertidumbre física.')
        if features.get('chem_efficiency', 0.0) < 0.5:
            recommendations.append('Explorar combustibles, mezclas o reacciones con mayor eficiencia antes de optimizar el diseño.')
        if features.get('math_complexity', 0.0) > 0.8:
            recommendations.append('Complementar la respuesta con validación matemática explícita de integrales, derivadas o matrices relevantes.')
        return {
            'prediction': round(prediction, 4),
            'research_intensity': research_intensity,
            'recommendations': recommendations[:4],
            'normalized': normalized,
        }
