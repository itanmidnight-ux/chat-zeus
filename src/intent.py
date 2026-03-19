"""Compatibility wrapper exposing legacy intent labels expected by existing tests."""
from __future__ import annotations

from src.utils.intent import classify_intent as _classify_intent

SIMPLE = 'simple'
EXPLICATIVE = 'explicativa'
ANALYTICAL = 'analitica'

_MAP = {
    'simple': SIMPLE,
    'explanatory': EXPLICATIVE,
    'analytical': ANALYTICAL,
}


def classify_intent(question: str) -> str:
    return _MAP.get(_classify_intent(question), SIMPLE)
