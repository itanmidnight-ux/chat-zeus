"""Problem decomposition into actionable subproblems."""
from __future__ import annotations

from src.utils.filters import clean_input

_DOMAIN_MAP = {
    'spacecraft': ['mission', 'propulsion', 'materials', 'energy', 'structure', 'safety'],
    'nave espacial': ['misión', 'propulsión', 'materiales', 'energía', 'estructura', 'seguridad'],
    'cohete': ['propulsión', 'combustible', 'control', 'estructura', 'seguridad'],
    'rocket': ['propulsion', 'fuel', 'control', 'structure', 'safety'],
    'ia': ['objetivo', 'datos', 'modelo', 'evaluación', 'despliegue'],
    'ai': ['objective', 'data', 'model', 'evaluation', 'deployment'],
}


def decompose_problem(question: str) -> list[str]:
    try:
        text = clean_input(question).lower()
        for keyword, tasks in _DOMAIN_MAP.items():
            if keyword in text:
                return tasks
        generic: list[str] = []
        if any(token in text for token in ('diseña', 'disena', 'design', 'crear', 'construir', 'build')):
            generic.extend(['objetivo', 'restricciones', 'componentes', 'riesgos', 'validación'])
        if any(token in text for token in ('explica', 'como', 'cómo', 'explain', 'why', 'por qué', 'porque')):
            generic.extend(['conceptos base', 'mecanismo', 'ejemplo'])
        if any(token in text for token in ('analiza', 'optimize', 'optimiza', 'evalúa', 'evalua', 'analyze')):
            generic.extend(['variables', 'hipótesis', 'escenarios', 'criterios'])
        deduped: list[str] = []
        for item in generic or ['contexto', 'objetivo', 'restricciones', 'solución']:
            if item not in deduped:
                deduped.append(item)
        return deduped
    except Exception:
        return ['contexto', 'objetivo', 'restricciones', 'solución']
