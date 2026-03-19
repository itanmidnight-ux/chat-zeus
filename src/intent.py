"""Clasificador ligero de intención."""
from __future__ import annotations

from src.utils import sanitize_text


SIMPLE = 'simple'
EXPLICATIVE = 'explicativa'
ANALYTICAL = 'analitica'


def classify_intent(question: str) -> str:
    text = sanitize_text(question).lower()
    if any(token in text for token in ('diseña', 'disena', 'analiza', 'optimiza', 'arquitectura', 'plan')):
        return ANALYTICAL
    if any(token in text for token in ('cómo', 'como', 'explica', 'qué es', 'que es', 'funciona')):
        return EXPLICATIVE
    if len(text.split()) <= 4 or any(token in text for token in ('hora', 'fecha', 'hola')):
        return SIMPLE
    return ANALYTICAL if len(text.split()) > 12 else EXPLICATIVE
