"""Generación de reportes JSON y texto legible para el usuario."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils import utc_now_iso, write_json


class ReportWriter:
    def __init__(self, report_dir: Path):
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def save(self, question: str, payload: dict[str, Any]) -> Path:
        filename = f"report_{payload.get('simulation', {}).get('run_id', 'session')}_{utc_now_iso().replace(':', '-')} .json".replace(' ', '')
        path = self.report_dir / filename
        write_json(path, {'question': question, **payload})
        return path

    def render_text(self, payload: dict[str, Any]) -> str:
        simulation = payload['simulation']
        ml = payload['ml']
        external = payload['external']
        calculations = payload.get('calculations', {})
        chemistry = simulation.get('chemistry', {})
        synthesis = external.get('synthesis', {})
        lines = [
            'Análisis completo del problema:',
            payload['analysis'],
            '',
        ]
        if simulation.get('mode') == 'general_analysis':
            lines.extend([
                'Marco analítico general:',
                f"- Tipo de problema: {simulation.get('problem_type', 'n/d')}",
                f"- Dominios analizados: {', '.join(simulation.get('domains', [])) or 'n/d'}",
                f"- Profundidad analítica: {simulation.get('analytical_depth', 'n/d')}",
                f"- Índice de incertidumbre: {simulation.get('uncertainty_index', 'n/d')}",
                f"- Anchura del análisis: {simulation.get('breadth_score', 'n/d')}",
                f"- Ejes de decisión: {', '.join(simulation.get('decision_axes', [])) or 'n/d'}",
                '',
                'Vacíos inmediatos a validar:',
                *[f"- {item}" for item in simulation.get('questions_to_validate', [])[:5]],
                '',
            ])
        else:
            lines.extend([
                'Variables consideradas:',
                f"- Masa útil: {simulation['payload_mass_kg']} kg",
                f"- Delta-v estimado: {simulation['delta_v_m_s']} m/s",
                f"- Altitud máxima simulada: {simulation['max_altitude_m']} m",
                f"- Alcance simplificado: {simulation['range_m']} m",
                f"- Tiempo de combustión efectivo: {simulation['burn_time_s']} s",
                f"- Eficiencia química estimada: {chemistry.get('estimated_efficiency', 'n/d')}",
                f"- Ejecución reanudada desde checkpoint: {'sí' if simulation.get('resource_profile', {}).get('resumed_from_checkpoint') else 'no'}",
                '',
                'Resultados y cálculos relevantes:',
                f"La simulación ligera por tareas pequeñas predice una velocidad final de {simulation['final_velocity_m_s']} m/s y combustible remanente de {simulation['remaining_fuel_kg']} kg.",
                f"El análisis termoquímico simplificado estima una presión efectiva de cámara de {chemistry.get('effective_pressure_pa', 'n/d')} Pa y un índice térmico de {chemistry.get('thermal_index', 'n/d')}.",
                f"El perfil de recursos mantuvo chunk_size={simulation.get('resource_profile', {}).get('chunk_size', 'n/d')} y un tope lógico de {simulation.get('resource_profile', {}).get('max_memory_mb', 'n/d')} MB por tarea.",
                '',
            ])
        lines.extend([
            '',
            'Hipótesis y predicción final:',
            f"- Predicción derivada: {ml['prediction']:.3f}",
            f"- Confianza estimada: {ml['confidence']:.2f}",
            f"- Intensidad de investigación sugerida por ML: {ml.get('research_intensity', 'n/d')} búsquedas",
            f"- Fuentes priorizadas por ML: {', '.join(ml.get('preferred_domains', [])) or 'n/d'}",
            f"- Pesos actuales por fuente: {ml.get('source_weights', {})}",
            f"- Estado interno del modelo: {ml.get('model_state', {})}",
            *[f'- {item}' for item in ml['hypotheses']],
            '',
            'Síntesis de investigación externa:',
            f"- Estado: {external['status']}",
            f"- Dominios detectados: {', '.join(external.get('domains', [])) or 'n/d'}",
            f"- Intenciones cubiertas: {', '.join(external.get('intents', [])) or 'n/d'}",
            f"- Consultas ejecutadas: {external.get('queries_executed', 'n/d')}",
            f"- Mezcla de fuentes: {external.get('sources_consulted', {})}",
            f"- Factibilidad aproximada: {synthesis.get('feasibility_signal', 'n/d')}",
            f"- Calidad media de evidencia: {synthesis.get('quality_score', 'n/d')}",
            f"- Fuente destacada: {external['source']}",
            f"- Extracto útil: {external['excerpt']}",
            f"- Perfil de conectividad: {synthesis.get('connectivity_profile', {})}",
        ])
        if synthesis.get('contradictions'):
            lines.extend(['', 'Contradicciones o cautelas detectadas:'])
            lines.extend([f'- {item}' for item in synthesis['contradictions'][:4]])
        if synthesis.get('research_gaps'):
            lines.extend(['', 'Vacíos de investigación:'])
            lines.extend([f'- {item}' for item in synthesis['research_gaps'][:5]])
        if synthesis.get('recommended_actions'):
            lines.extend(['', 'Siguientes acciones recomendadas:'])
            lines.extend([f'- {item}' for item in synthesis['recommended_actions'][:5]])
        if external.get('findings'):
            lines.extend(['', 'Hallazgos externos priorizados:'])
            for item in external['findings'][:6]:
                lines.append(
                    f"- [{item.get('source_type', 'fuente')}/{item.get('intent', 'general')}] score={item.get('score', 'n/d')} | {item.get('title')}: {item.get('snippet')} ({item.get('source')})"
                )
        if payload.get('optimization'):
            optimization = payload['optimization']
            best_result = optimization.get('best_result', {})
            lines.extend([
                '',
                'Optimización iterativa:',
                f"- Iteraciones: {optimization.get('iterations')}",
                f"- Objetivo: {optimization.get('objective')}",
                f"- Mejor puntuación: {optimization.get('best_score')}",
                f"- Mejor altitud hallada: {best_result.get('max_altitude_m', 'n/d')} m",
                f"- Mejor delta-v hallado: {best_result.get('delta_v_m_s', 'n/d')} m/s",
            ])
        if calculations.get('items') or calculations.get('variables'):
            lines.extend([
                '',
                'Cálculo analítico complementario:',
                f"- Estado: {calculations.get('status', 'n/d')}",
                f"- Variables consideradas: {', '.join(calculations.get('variables', [])) or 'n/d'}",
            ])
            for item in calculations.get('items', [])[:6]:
                lines.append(f"- [{item.get('type', 'analysis')}] {item.get('title', 'Resultado')}: {item.get('summary', 'n/d')}")
        lines.extend(['', 'Recomendaciones y conclusiones:', payload['conclusions']])
        return '\n'.join(lines)
