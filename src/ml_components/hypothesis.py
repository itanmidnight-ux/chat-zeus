"""Evaluación de hipótesis científicas y cálculo de confianza."""
from __future__ import annotations

from statistics import mean
from typing import Any

from src.utils import clamp


class HypothesisEvaluator:
    def evaluate(
        self,
        *,
        question: str,
        simulation_result: dict[str, Any],
        prediction: float,
        reliability_score: float,
        source_weights: dict[str, float],
        history_targets: list[float],
        uncertainty_drivers: list[str],
    ) -> dict[str, Any]:
        chemistry = simulation_result.get('chemistry', {})
        mode = simulation_result.get('mode', 'simulation')
        hypotheses = [
            'La predicción mejora cuando las simulaciones previas y la evidencia científica convergen en tendencias compatibles.',
            'Las fuentes con mejor historial y menor latencia deben tener más peso que fuentes externas conflictivas.',
        ]
        if mode == 'general_analysis':
            hypotheses.append('Las consultas analíticas generales requieren priorizar soporte documental y complejidad matemática sobre telemetría física directa.')
        else:
            hypotheses.append('La combinación entre delta-v, arrastre, masa útil y eficiencia química domina la respuesta de la simulación ligera.')
        if chemistry.get('estimated_efficiency', 0.0) < 0.4:
            hypotheses.append('La baja eficiencia química actual limita la confianza y sugiere revisar combustible, mezcla o temperatura de cámara.')
        if 'integral' in question.lower() or 'matriz' in question.lower():
            hypotheses.append('La complejidad matemática de la consulta aumenta la incertidumbre y requiere validación adicional.')

        history_mean = mean(history_targets) if history_targets else prediction
        confidence = clamp(0.25 + reliability_score * 0.35 + min(len(history_targets), 24) * 0.015 - abs(prediction - history_mean) / max(history_mean + 1.0, 1.0) * 0.1, 0.18, 0.97)
        return {
            'hypotheses': hypotheses[:5],
            'confidence': round(confidence, 4),
            'uncertainty_drivers': uncertainty_drivers,
            'top_sources': [name for name, _ in sorted(source_weights.items(), key=lambda item: item[1], reverse=True)],
        }
