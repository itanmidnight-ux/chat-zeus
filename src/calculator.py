"""Cálculo analítico ligero para matemáticas, materiales y validaciones simbólicas."""
from __future__ import annotations

import ast
import importlib
import importlib.util
import math
import re
from typing import Any

from src.utils import clamp, sanitize_text


class AnalyticalCalculator:
    """Resuelve cálculos ligeros y genera métricas analíticas complementarias."""

    def __init__(self) -> None:
        self._sympy = importlib.import_module('sympy') if importlib.util.find_spec('sympy') else None

    def analyze(self, question: str, simulation: dict[str, Any], knowledge_summary: str = '') -> dict[str, Any]:
        lowered = question.lower()
        calculations: list[dict[str, Any]] = []
        calculations.extend(self._math_calculations(question))
        calculations.extend(self._materials_calculations(question, simulation))
        calculations.extend(self._geology_calculations(question))
        calculations.extend(self._consistency_calculations(question, simulation, knowledge_summary))
        return {
            'status': 'ok' if calculations else 'no_direct_calculation',
            'items': calculations[:8],
            'variables': self._variables_considered(question, simulation),
        }

    def _variables_considered(self, question: str, simulation: dict[str, Any]) -> list[str]:
        variables = ['dominio inferido', 'contexto textual', 'resumen local RAG']
        if simulation.get('delta_v_m_s') is not None:
            variables.extend(['delta_v_m_s', 'max_altitude_m', 'range_m', 'burn_time_s'])
        if simulation.get('chemistry'):
            variables.extend(['mixture_ratio', 'estimated_efficiency'])
        if any(token in question.lower() for token in ['matriz', 'determinante', 'integral', 'derivada', 'ecuación']):
            variables.extend(['expresión simbólica', 'matriz de entrada'])
        return variables

    def _math_calculations(self, question: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        sp = self._sympy
        if sp is None:
            if any(token in question.lower() for token in ['derivada', 'integral', 'matriz', 'ecuación', 'resuelve', 'resolver']):
                return [{
                    'type': 'math',
                    'title': 'Motor simbólico no disponible',
                    'summary': 'No se encontró SymPy en el entorno actual; instala sympy para activar derivadas, integrales, ecuaciones y matrices simbólicas en Termux o Linux.',
                }]
            return items
        derivative_match = re.search(r'derivada\s+de\s+(.+?)(?:\s+en\s+x\s*=\s*([-0-9\.,]+))?$', question, re.IGNORECASE)
        if derivative_match:
            expr_text = derivative_match.group(1).strip()
            x = sp.symbols('x')
            expr = sp.sympify(expr_text.replace('^', '**'))
            derivative = sp.diff(expr, x)
            summary = f"Derivada simbólica de {expr_text}: {sp.simplify(derivative)}"
            if derivative_match.group(2):
                point = float(derivative_match.group(2).replace(',', '.'))
                summary += f"; evaluada en x={point}: {float(derivative.subs(x, point)):.6g}"
            items.append({'type': 'math', 'title': 'Derivada', 'summary': sanitize_text(summary)})

        integral_match = re.search(
            r'integral\s+de\s+(.+?)(?:\s+entre\s+([-0-9\.,]+)\s+y\s+([-0-9\.,]+))?$',
            question,
            re.IGNORECASE,
        )
        if integral_match:
            expr_text = integral_match.group(1).strip()
            x = sp.symbols('x')
            expr = sp.sympify(expr_text.replace('^', '**'))
            if integral_match.group(2) and integral_match.group(3):
                a = float(integral_match.group(2).replace(',', '.'))
                b = float(integral_match.group(3).replace(',', '.'))
                value = sp.integrate(expr, (x, a, b))
                summary = f"Integral definida de {expr_text} entre {a} y {b}: {sp.N(value, 8)}"
            else:
                value = sp.integrate(expr, x)
                summary = f"Integral indefinida de {expr_text}: {sp.simplify(value)}"
            items.append({'type': 'math', 'title': 'Integral', 'summary': sanitize_text(summary)})

        matrix_match = re.search(r'matriz\s+(\[\[.+\]\])', question, re.IGNORECASE)
        if matrix_match:
            raw = matrix_match.group(1)
            matrix_data = ast.literal_eval(raw)
            matrix = sp.Matrix(matrix_data)
            summary = f"Matriz {matrix.shape[0]}x{matrix.shape[1]} con determinante {matrix.det()}"
            if matrix.rows == matrix.cols and matrix.det() != 0:
                summary += f" e inversa {matrix.inv()}"
            items.append({'type': 'math', 'title': 'Matriz', 'summary': sanitize_text(summary)})

        equation_match = re.search(r'(?:resuelve|resolver|solve)\s+(.+?=\s*.+)$', question, re.IGNORECASE)
        if equation_match:
            expr_text = equation_match.group(1).strip()
            x = sp.symbols('x')
            left, right = expr_text.split('=', 1)
            solutions = sp.solve(sp.Eq(sp.sympify(left.replace('^', '**')), sp.sympify(right.replace('^', '**'))), x)
            items.append({'type': 'math', 'title': 'Ecuación', 'summary': sanitize_text(f"Soluciones de {expr_text}: {solutions}")})
        return items

    def _materials_calculations(self, question: str, simulation: dict[str, Any]) -> list[dict[str, Any]]:
        lowered = question.lower()
        if not any(token in lowered for token in ['material', 'esfuerzo', 'stress', 'tensión', 'resistencia']):
            return []
        load = self._extract_number(question, r'(?:carga|load|fuerza)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)')
        area = self._extract_number(question, r'(?:secci[oó]n|area|área)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)')
        strength = self._extract_number(question, r'(?:l[ií]mite|yield|resistencia)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)')
        if load is None or area is None:
            return []
        stress = load / max(area, 1e-9)
        summary = f"Esfuerzo medio estimado = carga/área = {stress:.6g} unidades de presión."
        if strength:
            safety_factor = strength / max(stress, 1e-9)
            summary += f" Factor de seguridad aproximado = {safety_factor:.4g}."
        if simulation.get('max_altitude_m', 0.0):
            thermal_factor = clamp(1.0 + simulation.get('max_altitude_m', 0.0) / 100000.0, 1.0, 3.5)
            summary += f" Se recomienda corregir por entorno térmico y dinámico con factor preliminar {thermal_factor:.3f}."
        return [{'type': 'materials', 'title': 'Resistencia de materiales', 'summary': sanitize_text(summary)}]

    def _geology_calculations(self, question: str) -> list[dict[str, Any]]:
        lowered = question.lower()
        if not any(token in lowered for token in ['geolog', 'roca', 'estrato', 'sedimento', 'sismo']):
            return []
        density = self._extract_number(question, r'(?:densidad)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)')
        thickness = self._extract_number(question, r'(?:espesor)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)')
        gravity = 9.81
        if density is not None and thickness is not None:
            lithostatic = density * gravity * thickness
            return [{
                'type': 'geology',
                'title': 'Carga litostática',
                'summary': sanitize_text(f"Carga litostática aproximada = densidad·g·espesor = {lithostatic:.6g} Pa si la densidad está en kg/m³ y el espesor en m."),
            }]
        return [{
            'type': 'geology',
            'title': 'Análisis geológico',
            'summary': 'Se detectó una consulta geológica; para cuantificar presión litostática o estabilidad conviene indicar densidad, espesor, pendiente o aceleración sísmica.',
        }]

    def _consistency_calculations(self, question: str, simulation: dict[str, Any], knowledge_summary: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if simulation.get('delta_v_m_s', 0.0) and simulation.get('burn_time_s', 0.0):
            acceleration = simulation['delta_v_m_s'] / max(simulation['burn_time_s'], 1e-9)
            items.append({
                'type': 'consistency',
                'title': 'Aceleración media equivalente',
                'summary': sanitize_text(f"Aceleración media equivalente ≈ delta_v / tiempo_de_quema = {acceleration:.6g} m/s²."),
            })
        if 'delta_v =' in knowledge_summary.lower() and simulation.get('payload_mass_kg') is not None:
            items.append({
                'type': 'consistency',
                'title': 'Validación con fórmula local',
                'summary': 'La respuesta se contrastó con la ecuación de Tsiolkovski y con heurísticas de rendimiento almacenadas en la base local.',
            })
        if any(token in question.lower() for token in ['energía', 'energia']) and simulation.get('final_velocity_m_s', 0.0):
            mass = max(float(simulation.get('payload_mass_kg', 1.0)), 1.0)
            energy = 0.5 * mass * simulation['final_velocity_m_s'] ** 2
            items.append({
                'type': 'physics',
                'title': 'Energía cinética',
                'summary': sanitize_text(f"Energía cinética asociada a la masa útil = 0.5·m·v² ≈ {energy:.6g} J."),
            })
        return items

    def _extract_number(self, text: str, pattern: str) -> float | None:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return None
        return float(match.group(1).replace(',', '.'))
