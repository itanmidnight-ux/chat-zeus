"""Generación de reportes JSON y texto legible para el usuario."""
from __future__ import annotations

import json
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
        lines = [
            '=== Análisis completo ===',
            payload['analysis'],
            '',
            '=== Parámetros principales ===',
            f"payload_mass_kg={simulation['payload_mass_kg']}",
            f"delta_v_m_s={simulation['delta_v_m_s']}",
            f"max_altitude_m={simulation['max_altitude_m']}",
            f"range_m={simulation['range_m']}",
            f"burn_time_s={simulation['burn_time_s']}",
            '',
            '=== Hipótesis y predicción ===',
            f"Predicción derivada={ml['prediction']:.3f}, confianza={ml['confidence']:.2f}",
            *[f'- {item}' for item in ml['hypotheses']],
            '',
            '=== Consulta externa ===',
            f"estado={external['status']}",
            f"fuente={external['source']}",
            f"extracto={external['excerpt']}",
        ]
        if payload.get('optimization'):
            lines.extend([
                '',
                '=== Optimización iterativa ===',
                json.dumps(payload['optimization'], indent=2, ensure_ascii=False),
            ])
        return '\n'.join(lines)
