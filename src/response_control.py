"""Control inteligente de intención, compresión y salida final para respuestas al usuario."""
from __future__ import annotations

import re
from typing import Any

from src.utils import sanitize_text

QUESTION_TYPE_SIMPLE = 'simple'
QUESTION_TYPE_EXPLICATIVE = 'explicativa'
QUESTION_TYPE_ANALYTICAL = 'analitica'

_SIMPLE_PATTERNS = (
    'que hora es', 'qué hora es', 'hora actual', 'fecha de hoy', 'quien eres', 'quién eres',
    'capital de', 'cuanto es', 'cuánto es', 'define ', 'dime ',
)
_EXPLANATORY_PATTERNS = (
    'como funciona', 'cómo funciona', 'explica', 'qué es', 'que es', 'por que', 'por qué',
    'como se', 'cómo se', 'diferencia entre', 'resume',
)
_ANALYTICAL_PATTERNS = (
    'diseña', 'disena', 'analiza', 'optimiza', 'plan', 'estrategia', 'simula', 'calcula',
    'dimensiona', 'arquitectura', 'propuesta', 'viabilidad', 'evalua', 'evalúa',
)

_INTERNAL_PATTERNS = [
    r'an[aá]lisis completo del problema:?',
    r'hip[oó]tesis(?: y predicci[oó]n final)?:?',
    r'ml weights?',
    r'\brag\b',
    r'fuentes(?: t[eé]cnicas)?',
    r'\blogs?\b',
    r'checkpoints?',
    r'https?://\S+',
    r'www\.\S+',
    r'\b(?:score|latency|quality_score|confidence|reliability|research_intensity|chunk_size|max_memory_mb|cpu_budget)\s*[=:]\s*[^,;\n]+',
]


def detect_question_type(question: str) -> str:
    text = sanitize_text(question).lower()
    if not text:
        return QUESTION_TYPE_SIMPLE
    if any(pattern in text for pattern in _ANALYTICAL_PATTERNS):
        return QUESTION_TYPE_ANALYTICAL
    if any(pattern in text for pattern in _EXPLANATORY_PATTERNS):
        return QUESTION_TYPE_EXPLICATIVE
    if len(text.split()) <= 6 or any(pattern in text for pattern in _SIMPLE_PATTERNS):
        return QUESTION_TYPE_SIMPLE
    if any(token in text for token in ('compar', 'riesgo', 'impacto', 'causa', 'funciona')):
        return QUESTION_TYPE_EXPLICATIVE
    return QUESTION_TYPE_ANALYTICAL if len(text.split()) > 18 else QUESTION_TYPE_EXPLICATIVE


def clean_output(raw_analysis: str) -> str:
    cleaned = raw_analysis
    for pattern in _INTERNAL_PATTERNS:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\([^)]*https?://[^)]*\)', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip(' .\n\t')


def summarize_intelligently(data: Any, level: str) -> str:
    if isinstance(data, dict):
        prioritized_keys = [
            'direct_answer', 'summary', 'conclusions', 'analysis', 'design_summary', 'key_points',
            'recommended_actions', 'notable_risks', 'excerpt', 'knowledge_summary',
        ]
        pieces: list[str] = []
        for key in prioritized_keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                pieces.append(value)
            elif isinstance(value, list):
                pieces.extend(str(item) for item in value[:3] if str(item).strip())
        if not pieces:
            pieces = [str(value) for value in data.values() if isinstance(value, (str, int, float))]
        text = ' '.join(pieces)
    elif isinstance(data, list):
        text = ' '.join(str(item) for item in data)
    else:
        text = str(data)

    text = clean_output(sanitize_text(text))
    sentences = [segment.strip() for segment in re.split(r'(?<=[\.!?])\s+', text) if segment.strip()]

    unique_sentences: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_sentences.append(sentence)

    limits = {
        QUESTION_TYPE_SIMPLE: 2,
        QUESTION_TYPE_EXPLICATIVE: 4,
        QUESTION_TYPE_ANALYTICAL: 5,
    }
    selected = unique_sentences[: limits.get(level, 3)]
    summarized = ' '.join(selected) if selected else text
    if level == QUESTION_TYPE_SIMPLE:
        return summarized[:220].strip()
    if level == QUESTION_TYPE_EXPLICATIVE:
        return summarized[:520].strip()
    return summarized[:820].strip()


def _build_simple_response(question: str, analysis_data: dict[str, Any]) -> str:
    direct_answer = analysis_data.get('direct_answer') or analysis_data.get('summary') or analysis_data.get('conclusions') or analysis_data.get('analysis')
    return summarize_intelligently({'summary': direct_answer}, QUESTION_TYPE_SIMPLE)


def _build_explicative_response(analysis_data: dict[str, Any]) -> str:
    key_points = analysis_data.get('key_points', [])[:3]
    summary = summarize_intelligently({
        'summary': analysis_data.get('summary', ''),
        'key_points': key_points,
        'conclusions': analysis_data.get('conclusions', ''),
    }, QUESTION_TYPE_EXPLICATIVE)
    return summary


def _build_analytical_response(analysis_data: dict[str, Any]) -> str:
    sections: list[str] = []
    design_summary = analysis_data.get('design_summary')
    if design_summary:
        sections.append(f'Resumen: {design_summary}')
    if analysis_data.get('key_points'):
        sections.append('Puntos clave: ' + '; '.join(analysis_data['key_points'][:3]))
    if analysis_data.get('notable_risks'):
        sections.append('Riesgos: ' + '; '.join(analysis_data['notable_risks'][:2]))
    if analysis_data.get('recommended_actions'):
        sections.append('Siguiente paso: ' + analysis_data['recommended_actions'][0])
    if analysis_data.get('conclusions'):
        sections.append(analysis_data['conclusions'])
    return summarize_intelligently(' '.join(sections), QUESTION_TYPE_ANALYTICAL)


def generate_response_by_level(level: str, analysis_data: dict[str, Any]) -> str:
    if level == QUESTION_TYPE_SIMPLE:
        return _build_simple_response(analysis_data.get('question', ''), analysis_data)
    if level == QUESTION_TYPE_EXPLICATIVE:
        return _build_explicative_response(analysis_data)
    return _build_analytical_response(analysis_data)


def estimate_implicit_satisfaction(question_type: str, response_text: str) -> float:
    length = len(response_text.split())
    targets = {
        QUESTION_TYPE_SIMPLE: (4, 24),
        QUESTION_TYPE_EXPLICATIVE: (25, 90),
        QUESTION_TYPE_ANALYTICAL: (60, 180),
    }
    low, high = targets.get(question_type, (20, 80))
    if low <= length <= high:
        return 0.9
    distance = min(abs(length - low), abs(length - high))
    return round(max(0.35, 0.9 - distance / max(high, 1)), 3)


def build_user_response(question: str, analysis_data: dict[str, Any]) -> tuple[str, str]:
    level = detect_question_type(question)
    payload = dict(analysis_data)
    payload['question'] = question
    response = generate_response_by_level(level, payload)
    response = clean_output(response)
    response = sanitize_text(response)
    return level, response
