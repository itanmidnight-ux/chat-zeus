"""Descomposición ligera de problemas complejos en subproblemas accionables."""
from __future__ import annotations

from src.utils import sanitize_text

_DOMAIN_MAP = {
    'nave espacial': ['misión', 'propulsión', 'estructura', 'energía', 'materiales', 'seguridad'],
    'cohete': ['propulsión', 'combustible', 'control', 'estructura', 'operación'],
    'ia': ['objetivo', 'datos', 'modelo', 'evaluación', 'despliegue'],
    'negocio': ['objetivo', 'costes', 'riesgos', 'clientes', 'operación'],
}


def decompose_problem(question: str) -> list[str]:
    text = sanitize_text(question).lower()
    for keyword, tasks in _DOMAIN_MAP.items():
        if keyword in text:
            return tasks
    generic = []
    if any(token in text for token in ('diseña', 'disena', 'crear', 'construir')):
        generic.extend(['objetivo', 'restricciones', 'componentes', 'riesgos', 'validación'])
    if any(token in text for token in ('explica', 'como', 'cómo', 'por qué', 'porque')):
        generic.extend(['conceptos base', 'mecanismo', 'ejemplo'])
    if any(token in text for token in ('analiza', 'optimiza', 'evalúa', 'evalua')):
        generic.extend(['variables', 'hipótesis', 'escenarios', 'criterios'])
    return generic or ['contexto', 'objetivo', 'restricciones', 'solución']
