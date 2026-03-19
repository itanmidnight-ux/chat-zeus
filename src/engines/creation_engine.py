"""Creation engine for formulas, explanations, and practical solution text."""
from __future__ import annotations


def generate_solution(question: str) -> str:
    text = question.lower()
    if 'cateto' in text:
        return 'En un triángulo rectángulo: cateto = √(hipotenusa² - otro cateto²).'
    if 'pitag' in text:
        return 'Teorema de Pitágoras: hipotenusa² = cateto_a² + cateto_b².'
    if 'ohm' in text:
        return 'Ley de Ohm: V = I × R. También I = V / R y R = V / I.'
    if 'derive' in text or 'derivada' in text:
        return 'Regla básica: d/dx(x^n) = n·x^(n-1).'
    if 'explain' in text or 'explica' in text:
        return 'La respuesta útil es dividir el problema, identificar variables clave y validar el resultado con un ejemplo concreto.'
    return 'Puedo ayudarte con fórmulas, explicaciones técnicas, ideas de diseño y pasos prácticos si haces la petición más específica.'
