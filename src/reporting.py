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
        filename = f"report_{payload.get('simulation', {}).get('run_id', 'session')}_{utc_now_iso().replace(':', '-')}.json"
        path = self.report_dir / filename
        write_json(path, {'question': question, **payload})
        return path

    def render_text(self, payload: dict[str, Any]) -> str:
        simulation = payload['simulation']
        ml = payload['ml']
        external = payload['external']
        chemistry = simulation.get('chemistry', {})
        lines = [
            'Análisis completo del problema:',
            payload['analysis'],
            '',
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
            'Hipótesis y predicción final:',
            f"- Predicción derivada: {ml['prediction']:.3f}",
            f"- Confianza estimada: {ml['confidence']:.2f}",
            *[f'- {item}' for item in ml['hypotheses']],
            '',
            'Recomendaciones y conclusiones:',
            payload['conclusions'],
            '',
            'Complemento externo:',
            f"- Estado: {external['status']}",
            f"- Fuente: {external['source']}",
            f"- Extracto útil: {external['excerpt']}",
        ]
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
        return '\n'.join(lines)
