"""Semantic understanding helpers for low-memory autonomous routing."""
from __future__ import annotations

from dataclasses import dataclass
from math import log1p
import re

from src.core.decomposer import decompose_problem
from src.utils.filters import clean_input

_FACT_WORDS = {
    'who', 'what', 'when', 'where', 'capital', 'define', 'meaning', 'richest',
    'quién', 'quien', 'qué', 'que', 'cuál', 'cual', 'capital', 'dato', 'hecho',
}
_EXPLAIN_WORDS = {
    'how', 'why', 'compare', 'explain', 'analysis', 'analyze', 'reason', 'impact',
    'cómo', 'como', 'por qué', 'porque', 'explica', 'analiza', 'analisis', 'compara', 'impacto',
}
_CREATE_WORDS = {
    'design', 'build', 'create', 'generate', 'write', 'draft', 'plan',
    'diseña', 'disena', 'crea', 'genera', 'escribe', 'arquitectura', 'propuesta', 'formula', 'fórmula',
}
_EXECUTION_WORDS = {'python:', 'code:', 'run', 'execute', 'ejecuta', 'script', 'programa'}
_TIME_WORDS = {'time', 'hora'}
_DATE_WORDS = {'date', 'fecha', 'día', 'dia', 'today', 'hoy'}
_MATH_RE = re.compile(r'(?:(?:\d+[\d\s+\-*/().,^%]*)|(?:[+\-/*().%^\s]+\d)){3,}')
_ENTITY_RE = re.compile(r"\b[a-záéíóúñ0-9]{3,}\b", re.IGNORECASE)


@dataclass
class UnderstandingResult:
    normalized_question: str
    intent_scores: dict[str, float]
    selected_intent: str
    entities: list[str]
    tasks: list[str]
    requires_freshness: bool
    needs_multi_step: bool
    estimated_complexity: float
    route_candidates: list[str]
    pattern_key: str


class SemanticUnderstandingEngine:
    """Feature-light semantic understanding for broad user requests."""

    INTENTS = ('math', 'time', 'date', 'fact', 'creation', 'analysis', 'execution', 'simple')

    def analyze(self, question: str) -> UnderstandingResult:
        text = clean_input(question)
        tokens = text.split()
        scores = {intent: 0.05 for intent in self.INTENTS}
        joined = f' {text} '

        if not text:
            scores['simple'] = 0.9
        if self._looks_like_math(text):
            scores['math'] += 0.85
        if any(token in joined for token in (' python:', ' code:')) or any(word in text for word in _EXECUTION_WORDS):
            scores['execution'] += 0.8
        if any(word in text for word in _TIME_WORDS):
            scores['time'] += 0.85
        if any(word in text for word in _DATE_WORDS):
            scores['date'] += 0.85
        if any(word in text for word in _CREATE_WORDS):
            scores['creation'] += 0.55
        if any(word in text for word in _EXPLAIN_WORDS):
            scores['analysis'] += 0.5
        if any(word in text for word in _FACT_WORDS) or '?' in text:
            scores['fact'] += 0.45
        if len(tokens) <= 4:
            scores['simple'] += 0.35
        if len(tokens) >= 12:
            scores['analysis'] += 0.15
            scores['creation'] += 0.1

        if scores['time'] > 0.6:
            scores['simple'] += 0.2
        if scores['date'] > 0.6:
            scores['simple'] += 0.2
        if scores['math'] > 0.5:
            scores['analysis'] += 0.1

        selected_intent = max(scores, key=scores.get)
        tasks = decompose_problem(text)
        entities = self._extract_entities(text)
        needs_multi_step = len(tasks) > 2 or len(tokens) > 10 or selected_intent in {'analysis', 'creation'}
        requires_freshness = any(marker in text for marker in ('today', 'latest', 'recent', 'actual', 'actualidad', 'hoy', 'ahora'))
        complexity = round(min(1.0, log1p(len(tokens) + len(tasks)) / 3.2 + (0.15 if needs_multi_step else 0.0)), 3)
        route_candidates = self._rank_routes(scores)
        pattern_key = self._build_pattern_key(selected_intent, entities, tasks, requires_freshness)
        return UnderstandingResult(
            normalized_question=text,
            intent_scores={name: round(value, 4) for name, value in scores.items()},
            selected_intent=selected_intent,
            entities=entities,
            tasks=tasks,
            requires_freshness=requires_freshness,
            needs_multi_step=needs_multi_step,
            estimated_complexity=complexity,
            route_candidates=route_candidates,
            pattern_key=pattern_key,
        )

    def _rank_routes(self, scores: dict[str, float]) -> list[str]:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        routes: list[str] = []
        for intent, _ in ranked:
            route = 'retrieve'
            if intent == 'math':
                route = 'math'
            elif intent in {'time', 'date', 'simple'}:
                route = 'direct'
            elif intent == 'creation':
                route = 'create'
            elif intent == 'execution':
                route = 'execute'
            elif intent == 'analysis':
                route = 'analyze'
            if route not in routes:
                routes.append(route)
        return routes[:4]

    def _extract_entities(self, text: str) -> list[str]:
        common = {'para', 'como', 'qué', 'que', 'the', 'and', 'with', 'una', 'uno', 'las', 'los'}
        entities: list[str] = []
        for item in _ENTITY_RE.findall(text):
            if item in common or item.isdigit():
                continue
            if item not in entities:
                entities.append(item)
        return entities[:10]

    def _looks_like_math(self, text: str) -> bool:
        compact = text.replace(' ', '')
        symbolic = any(op in compact for op in ('+', '-', '*', '/', '%', '^')) and any(ch.isdigit() for ch in compact)
        return bool(_MATH_RE.fullmatch(text)) or symbolic or any(marker in text for marker in ('calculate', 'solve', 'cuánto es', 'cuanto es'))

    def _build_pattern_key(self, intent: str, entities: list[str], tasks: list[str], requires_freshness: bool) -> str:
        entity_part = '-'.join(entities[:2]) if entities else 'generic'
        task_part = '-'.join(task[:10] for task in tasks[:2]) if tasks else 'none'
        freshness = 'fresh' if requires_freshness else 'stable'
        return f'{intent}:{freshness}:{entity_part}:{task_part}'
