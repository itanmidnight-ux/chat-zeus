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
    if 'nave espacial' in text or 'spacecraft' in text:
        return (
            'Diseño base: separa misión, propulsión, estructura, energía y seguridad. '
            'Usa redundancia en control, blindaje moderado para instrumentos y un presupuesto térmico conservador. '
            'Valida masa, potencia disponible y riesgos antes de cerrar la arquitectura.'
        )
    if 'arquitectura' in text or 'architecture' in text:
        return (
            'Propuesta: define objetivos, restricciones, módulos principales, flujo de datos, verificación y estrategia de mejora continua. '
            'Prioriza componentes desacoplados, observabilidad y validación incremental.'
        )
    if 'explain' in text or 'explica' in text:
        return 'La respuesta útil es dividir el problema, identificar variables clave y validar el resultado con un ejemplo concreto.'
    return (
        'Plan sugerido: define el objetivo, enumera restricciones, divide el problema en componentes, '
        'evalúa riesgos y propone una validación corta antes de ejecutar cambios grandes.'
    )
