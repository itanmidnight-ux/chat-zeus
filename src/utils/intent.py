"""Rule-based lightweight intent classifier."""
from __future__ import annotations

from src.utils.filters import clean_input

_SIMPLE = ('time', 'date', 'hora', 'fecha', 'hola', 'hello', 'hi')
_EXPLANATORY = ('how', 'why', 'explain', 'qué es', 'que es', 'como', 'cómo', 'funciona', 'explica')
_ANALYTICAL = ('design', 'analyze', 'analyse', 'optimize', 'plan', 'strategy', 'build', 'diseña', 'disena', 'analiza', 'optimiza', 'arquitectura')


def classify_intent(question: str) -> str:
    try:
        text = clean_input(question).lower()
        if any(token in text for token in _ANALYTICAL):
            return 'analytical'
        if any(token in text for token in _EXPLANATORY):
            return 'explanatory'
        if len(text.split()) <= 5 or any(token in text for token in _SIMPLE):
            return 'simple'
        return 'analytical' if len(text.split()) > 12 else 'explanatory'
    except Exception:
        return 'simple'
