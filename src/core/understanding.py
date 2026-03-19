"""Semantic understanding helpers for low-memory autonomous routing."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from math import log1p
import re

from src.core.decomposer import decompose_problem
from src.utils.filters import clean_input

_FACT_WORDS = {
    'who', 'what', 'when', 'where', 'which', 'capital', 'define', 'meaning', 'richest',
    'quién', 'quien', 'qué', 'que', 'cuál', 'cual', 'capital', 'dato', 'hecho', 'explica',
}
_CREATE_WORDS = {
    'design', 'build', 'create', 'generate', 'write', 'draft', 'plan', 'system', 'architecture',
    'diseña', 'disena', 'crea', 'genera', 'escribe', 'arquitectura', 'propuesta', 'formula', 'fórmula',
}
_CONVERSATION_WORDS = {
    'hola', 'hello', 'hi', 'hey', 'thanks', 'gracias', 'good morning', 'good afternoon', 'buenas',
    'how are you', 'como estas', 'cómo estás', 'nice to meet you', 'encantado',
}
_IDENTITY_WORDS = {'who are you', 'quien eres', 'quién eres', 'your name', 'tu nombre'}
_TIME_WORDS = {'qué hora es', 'que hora es', 'hora actual', 'current time', 'time now'}
_DATE_WORDS = {'qué fecha es', 'que fecha es', 'fecha de hoy', 'hoy es', 'current date', "today's date", 'date today'}
_MEMORY_WORDS = {'cómo me llamo', 'como me llamo', 'cuál es mi nombre', 'cual es mi nombre', 'what is my name'}
_CLARIFY_CUES = {'help', 'ayuda', 'something', 'algo', 'this', 'esto', 'that', 'eso'}
_MATH_RE = re.compile(r'(?:(?:\d+[\d\s+\-*/().,^%]*)|(?:[+\-/*().%^\s]+\d)){3,}')
_ENTITY_RE = re.compile(r"\b[a-záéíóúñ0-9]{2,}\b", re.IGNORECASE)


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
    ambiguity_score: float
    inferred_profile: dict[str, str]


class SemanticUnderstandingEngine:
    """Intent inference that tolerates imperfect input and extracts lightweight user context."""

    INTENTS = ('conversation', 'math', 'fact', 'creation', 'identity', 'time', 'date', 'clarification_needed', 'analysis', 'execution', 'simple')

    def analyze(self, question: str) -> UnderstandingResult:
        text = clean_input(question)
        tokens = text.split()
        scores = {intent: 0.03 for intent in self.INTENTS}
        joined = f' {text} '

        if not text:
            scores['clarification_needed'] = 0.95

        if self._looks_like_conversation(text):
            scores['conversation'] += 0.78
        if self._looks_like_math(text):
            scores['math'] += 0.85
        if any(word in joined for word in (f' {w} ' for w in _CREATE_WORDS)):
            scores['creation'] += 0.65
        identity_detected = any(phrase in text for phrase in _IDENTITY_WORDS)
        if identity_detected:
            scores['identity'] += 0.88
            scores['conversation'] = max(scores['conversation'] - 0.12, 0.03)
        if any(phrase in text for phrase in _TIME_WORDS):
            scores['time'] += 0.96
            scores['simple'] += 0.35
        if any(phrase in text for phrase in _DATE_WORDS):
            scores['date'] += 0.96
            scores['simple'] += 0.35
        if any(phrase in text for phrase in _MEMORY_WORDS):
            scores['identity'] += 0.82
            scores['simple'] += 0.28
        if any(word in joined for word in (f' {w} ' for w in _FACT_WORDS)) or '?' in text:
            scores['fact'] += 0.48
        if any(word in text for word in ('python:', 'code:', 'run ', 'execute ', 'script')):
            scores['execution'] += 0.88
        if len(tokens) >= 10:
            scores['analysis'] += 0.2
            scores['creation'] += 0.12
        if len(tokens) <= 3:
            scores['conversation'] += 0.15
            scores['simple'] += 0.24

        profile = self._extract_profile(text)
        if profile:
            scores['conversation'] += 0.52
            scores['identity'] += 0.12

        ambiguity_score = self._ambiguity_score(text, tokens)
        if ambiguity_score > 0.45:
            scores['clarification_needed'] += ambiguity_score
        if scores['math'] > 0.4:
            scores['analysis'] += 0.12
        if scores['conversation'] > 0.6:
            scores['simple'] += 0.16

        selected_intent = max(scores, key=scores.get)
        tasks = decompose_problem(text)
        entities = self._extract_entities(text)
        needs_multi_step = len(tasks) > 2 or len(tokens) > 12 or selected_intent in {'analysis', 'creation'}
        requires_freshness = any(marker in text for marker in ('today', 'latest', 'recent', 'actual', 'actualidad', 'hoy', 'ahora'))
        complexity = round(min(1.0, log1p(len(tokens) + len(tasks)) / 3.1 + (0.12 if needs_multi_step else 0.0)), 3)
        route_candidates = self._rank_routes(scores)
        pattern_key = self._build_pattern_key(selected_intent, entities, tasks, requires_freshness, profile)
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
            ambiguity_score=round(ambiguity_score, 4),
            inferred_profile=profile,
        )

    def _rank_routes(self, scores: dict[str, float]) -> list[str]:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        routes: list[str] = []
        for intent, _ in ranked:
            route = 'retrieve'
            if intent == 'conversation':
                route = 'converse'
            elif intent == 'math':
                route = 'math'
            elif intent in {'identity', 'simple'}:
                route = 'direct'
            elif intent == 'clarification_needed':
                route = 'clarify'
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
        common = {'para', 'como', 'qué', 'que', 'the', 'and', 'with', 'una', 'uno', 'las', 'los', 'por', 'favor'}
        entities: list[str] = []
        for item in _ENTITY_RE.findall(text):
            if item in common or item.isdigit():
                continue
            if item not in entities:
                entities.append(item)
        return entities[:12]

    def _extract_profile(self, text: str) -> dict[str, str]:
        profile: dict[str, str] = {}
        name_match = re.search(r'\b(?:i am|im|my name is|me llamo|soy)\s+([a-záéíóúñ]{2,})', text)
        if name_match:
            profile['name'] = name_match.group(1).title()
        like_match = re.search(r'\b(?:i like|me gusta|prefiero)\s+([^,.!?]+)', text)
        if like_match:
            profile['preference'] = like_match.group(1).strip()[:80]
        return profile

    def _looks_like_conversation(self, text: str) -> bool:
        if any(phrase in text for phrase in _CONVERSATION_WORDS):
            return True
        compact = text.replace('?', '').replace('!', '').strip()
        return any(SequenceMatcher(None, compact, sample).ratio() > 0.78 for sample in ('hola', 'hello', 'how are you', 'gracias'))

    def _looks_like_math(self, text: str) -> bool:
        compact = text.replace(' ', '')
        symbolic = any(op in compact for op in ('+', '-', '*', '/', '%', '^')) and any(ch.isdigit() for ch in compact)
        math_words = ('calculate', 'solve', 'cuánto es', 'cuanto es', 'suma', 'resta', 'multiplica', 'divide')
        return bool(_MATH_RE.fullmatch(text)) or symbolic or any(marker in text for marker in math_words)

    def _ambiguity_score(self, text: str, tokens: list[str]) -> float:
        if not text:
            return 1.0
        score = 0.0
        if len(tokens) <= 2 and not self._looks_like_conversation(text):
            score += 0.35
        if text.endswith(('?', '¿')) and len(tokens) <= 4:
            score += 0.15
        if any(token in _CLARIFY_CUES for token in tokens):
            score += 0.18
        if any(token in {'it', 'this', 'that', 'eso', 'esto'} for token in tokens) and len(tokens) <= 6:
            score += 0.22
        if re.search(r'\b(and|y|with|con)\b\s*$', text):
            score += 0.4
        return min(1.0, score)

    def _build_pattern_key(self, intent: str, entities: list[str], tasks: list[str], requires_freshness: bool, profile: dict[str, str]) -> str:
        entity_part = '-'.join(entities[:2]) if entities else 'generic'
        task_part = '-'.join(task[:10] for task in tasks[:2]) if tasks else 'none'
        freshness = 'fresh' if requires_freshness else 'stable'
        profile_part = 'profile' if profile else 'anonymous'
        return f'{intent}:{freshness}:{profile_part}:{entity_part}:{task_part}'
