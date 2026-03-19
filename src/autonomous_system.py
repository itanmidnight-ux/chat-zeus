"""Autonomous reasoning system implementing a low-memory learning pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
import time
from typing import Any

from src.agents.memory import MemoryAgent
from src.cognitive_system import CognitiveSystem
from src.config import CONFIG
from src.core.confidence import ConfidenceEngine
from src.core.decision import DecisionEngine
from src.core.episodic import EpisodeLearner
from src.core.executor import TaskExecutor
from src.core.learning import LearningEngine
from src.core.memory import LightweightMemory
from src.core.understanding import SemanticUnderstandingEngine, UnderstandingResult
from src.core.verifier import VerificationEngine
from src.engines.fact_engine import FactEngine
from src.storage import StorageManager
from src.utils.filters import clean_input, clean_output


@dataclass
class AutonomousResult:
    question: str
    intent: str
    tasks: list[str]
    plan: list[dict[str, Any]]
    research: dict[str, Any]
    best_solution: dict[str, Any]
    memory: dict[str, Any]
    response_text: str


class SessionContext:
    def __init__(self, memory_store: LightweightMemory, max_turns: int = 8):
        self.memory_store = memory_store
        self.max_turns = max_turns
        self.profile: dict[str, str] = {'assistant_name': 'Chat Zeus'}
        self.turns: list[dict[str, str]] = []

    def update_from_understanding(self, understanding: UnderstandingResult) -> None:
        for key, value in understanding.inferred_profile.items():
            self.profile[key] = value
            self.memory_store.put('facts', f'user_profile:{key}', {'query': key, 'value': value, 'source': 'session'})

    def register_turn(self, question: str, response: str) -> None:
        self.turns.append({'question': question, 'response': response})
        self.turns = self.turns[-self.max_turns :]

    def recall(self, question: str) -> dict[str, str]:
        context = dict(self.profile)
        if self.turns:
            context['last_user_message'] = self.turns[-1]['question']
            context['last_response'] = self.turns[-1]['response']
        cached_name = self.memory_store.get('facts', 'user_profile:name')
        if cached_name and 'name' not in context:
            context['name'] = str(cached_name.get('value', ''))
        return context


class SimpleIntentRouter:
    """Strict low-latency router for intents that must bypass the full pipeline."""

    _GREETING_RE = re.compile(r"^(hola|buenas|hello|hi|hey)(?:[!. ]*)$")
    _TIME_RE = re.compile(r"\b(?:qué hora es|que hora es|hora actual|current time|time now)\b")
    _DATE_RE = re.compile(r"\b(?:qué fecha es|que fecha es|fecha de hoy|hoy es|current date|today'?s date|date today)\b")
    _IDENTITY_RE = re.compile(r"\b(?:quién eres|quien eres|tu nombre|your name|who are you)\b")
    _MEMORY_NAME_RE = re.compile(r"\b(?:cómo me llamo|como me llamo|cuál es mi nombre|cual es mi nombre|what is my name)\b")

    def route(self, question: str, context: dict[str, str], memory_store: LightweightMemory) -> tuple[str, str] | None:
        text = clean_input(question)
        if not text:
            return 'conversation', 'Hola. ¿En qué puedo ayudarte?'
        lowered = text.lower()
        if self._GREETING_RE.search(lowered):
            name = context.get('name')
            return 'conversation', f'Hola{", " + name if name else ""}. ¿En qué puedo ayudarte?'
        if self._TIME_RE.search(lowered):
            return 'time', f'Son las {self.get_time()}.'
        if self._DATE_RE.search(lowered):
            return 'date', f'La fecha es {self.get_date()}.'
        if self._IDENTITY_RE.search(lowered):
            assistant_name = context.get('assistant_name', 'Chat Zeus')
            return 'identity', f'Soy {assistant_name}, un asistente científico con memoria de sesión y razonamiento estructurado.'
        if self._MEMORY_NAME_RE.search(lowered):
            stored = context.get('name') or self._memory_name(memory_store)
            if stored:
                return 'identity', f'Te llamas {stored}.'
            return 'identity', 'Aún no me has dicho tu nombre.'
        return None

    @staticmethod
    def get_time() -> str:
        return datetime.now().strftime('%H:%M:%S')

    @staticmethod
    def get_date() -> str:
        return datetime.now().strftime('%Y-%m-%d')

    @staticmethod
    def _memory_name(memory_store: LightweightMemory) -> str | None:
        item = memory_store.get('facts', 'user_profile:name')
        if not item:
            return None
        value = str(item.get('value', '')).strip()
        return value or None


class ResponseFinalizer:
    """Produce only final user-facing answers and reject noisy/unrelated text."""

    _NOISE_PATTERNS = (
        'confianza estimada', 'análisis completo', 'analysis:', 'plan:',
        'logs', 'checkpoints', 'ml weights', 'random fact', 'artículo relacionado', 'related article',
    )

    def finalize(self, question: str, response: str, *, fallback: str | None = None) -> str:
        cleaned = clean_output(response or '')
        cleaned = re.sub(r'\bConfianza estimada:[^.]+\.?', ' ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if self._is_unrelated(question, cleaned):
            cleaned = clean_output(fallback or '')
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if self._is_unrelated(question, cleaned):
            return 'No pude generar una respuesta final útil con la información disponible.'
        return cleaned or 'No pude generar una respuesta final útil con la información disponible.'

    def _is_unrelated(self, question: str, response: str) -> bool:
        if not response:
            return True
        lowered = response.lower()
        if any(marker in lowered for marker in self._NOISE_PATTERNS):
            return True
        q_tokens = {token for token in re.findall(r'[a-záéíóúñ0-9]{3,}', question.lower())}
        r_tokens = {token for token in re.findall(r'[a-záéíóúñ0-9]{3,}', lowered)}
        if not q_tokens:
            return False
        overlap = q_tokens & r_tokens
        simple_queries = {'hola', 'hora', 'fecha', 'nombre', 'llamo', 'eres'}
        if overlap:
            return False
        return len(q_tokens - simple_queries) >= 2


class ClarificationEngine:
    def needs_clarification(self, understanding: UnderstandingResult, question: str) -> str | None:
        text = question.strip()
        if understanding.selected_intent == 'clarification_needed' or understanding.ambiguity_score > 0.55:
            return 'Necesito un poco más de contexto para ayudarte bien. ¿Qué objetivo exacto quieres lograr?'
        if understanding.selected_intent == 'math' and not any(ch.isdigit() for ch in text):
            return 'Veo una solicitud matemática, pero faltan números o una expresión concreta. ¿Cuál es la operación completa?'
        if understanding.selected_intent == 'creation' and len(understanding.entities) < 2 and len(text.split()) < 6:
            return 'Puedo diseñarlo, pero necesito más detalle. ¿Qué sistema quieres crear y bajo qué restricciones?'
        if understanding.selected_intent == 'fact' and any(token in text for token in ('it', 'this', 'that', 'eso', 'esto')) and len(text.split()) < 7:
            return '¿A qué tema u objeto te refieres exactamente?'
        return None


class AutonomousReasoningSystem:
    def __init__(self, storage: StorageManager | None = None, memory_agent: MemoryAgent | None = None, *_: Any, memory_path=None, **__: Any):
        self.storage = storage or StorageManager(CONFIG.db_path, CONFIG.checkpoint_dir)
        path = memory_path or (memory_agent.path if memory_agent is not None else (CONFIG.models_dir / 'agent_memory.json'))
        self.memory_store = LightweightMemory(path)
        self.memory_agent = memory_agent or MemoryAgent(self.storage, path)
        self.learning_engine = LearningEngine(self.memory_store, storage=self.storage, timeout=CONFIG.internet_timeout_sec)
        self.fact_engine = FactEngine(self.learning_engine)
        self.executor = TaskExecutor(self.fact_engine)
        self.understanding = SemanticUnderstandingEngine()
        self.decision_engine = DecisionEngine(max_memory_mb=CONFIG.max_task_memory_mb, max_external_queries=CONFIG.max_external_queries)
        self.verifier = VerificationEngine()
        self.confidence_engine = ConfidenceEngine()
        self.episode_learner = EpisodeLearner(self.storage)
        self.session_context = SessionContext(self.memory_store)
        self.simple_router = SimpleIntentRouter()
        self.finalizer = ResponseFinalizer()
        self.clarifier = ClarificationEngine()
        self.cognitive_system = CognitiveSystem(
            storage=self.storage,
            long_term_memory=self.memory_store,
            task_executor=self.executor,
            learning_engine=self.learning_engine,
            max_history=CONFIG.max_history_messages,
        )

    def _memory_lookup(self, understanding: UnderstandingResult) -> dict[str, Any] | None:
        for bucket in ('facts', 'solutions', 'patterns'):
            hit = self.memory_store.get(bucket, understanding.normalized_question)
            if hit:
                return {**hit, 'bucket': bucket}
        learned = self.storage.load_learned_pattern(understanding.pattern_key)
        if learned:
            return {'bucket': 'patterns', 'value': learned.get('sample_question', ''), 'source': 'strategy'}
        return None

    def _execute_plan(self, engine: str, question: str, context: dict[str, str]) -> tuple[str, list[str]]:
        sources: list[str] = []
        if engine == 'analysis':
            local_hits = self.storage.search_knowledge(question, limit=3)
            if local_hits:
                summary = ' '.join(item['content'] for item in local_hits[:2])
                sources.extend([str(item['source']) for item in local_hits[:2]])
                response = self.executor.execute_task('analysis', f'{question}. contexto: {summary}', context=context)
                return response, sources
        response = self.executor.execute_task(engine, question, context=context)
        if engine == 'fact':
            cached = self.memory_store.get('facts', question)
            if cached:
                sources.append(str(cached.get('source', 'memory')))
        return response, sources

    def process(self, question: str) -> AutonomousResult:
        started = time.perf_counter()
        cleaned = clean_input(question)
        understanding = self.understanding.analyze(cleaned)
        self.session_context.update_from_understanding(understanding)
        context = self.session_context.recall(cleaned)

        direct_route = self.simple_router.route(cleaned, context, self.memory_store)
        if direct_route is not None:
            intent, response = direct_route
            response = self.finalizer.finalize(cleaned, response)
            self.memory_store.put('facts', cleaned, response, source='simple_router')
            self.session_context.register_turn(cleaned, response)
            return AutonomousResult(
                question=cleaned,
                intent=intent,
                tasks=[intent],
                plan=[{'step': 'route_simple_intent', 'status': 'completed'}],
                research={'used_learning': False, 'sources': [], 'requires_freshness': False},
                best_solution={'task': intent, 'proposal': response, 'final_score': 0.99, 'route': 'simple_router'},
                memory=self.memory_store.export(),
                response_text=response,
            )

        cognitive_result = self.cognitive_system.process(cleaned)

        clarification = self.clarifier.needs_clarification(understanding, cleaned)
        if clarification:
            self.session_context.register_turn(cleaned, clarification)
            return AutonomousResult(
                question=cleaned,
                intent='clarification_needed',
                tasks=understanding.tasks or ['clarify'],
                plan=[{'step': 'understand', 'status': 'completed'}, {'step': 'clarify', 'status': 'completed'}],
                research={'used_learning': False, 'sources': [], 'requires_freshness': False},
                best_solution={'task': 'clarification', 'proposal': clarification, 'final_score': 0.92},
                memory=self.memory_store.export(),
                response_text=clarification,
            )

        memory_hit = self._memory_lookup(understanding)
        decision = self.decision_engine.decide(understanding, hot_memory=memory_hit)
        route = decision.route
        engine = 'conversation' if understanding.selected_intent == 'conversation' else decision.engine
        response, sources = self._execute_plan(engine, cleaned, context)
        if cognitive_result.response_text and understanding.selected_intent in {'creation', 'analysis', 'clarification_needed'}:
            response = cognitive_result.response_text
            sources = list({*sources, *[item.get('source', 'local') for item in cognitive_result.research.get('findings', []) if isinstance(item, dict)]})
        if not response and route != 'retrieve':
            response = self.learning_engine.search_and_learn(cleaned)
            sources.append('internet')
        response = clean_output(response or 'No pude completar la solicitud con suficiente confianza.')
        verification = self.verifier.verify(cleaned, response, source_count=len(sources), executed=engine == 'execution')
        if verification.score < 0.42 and understanding.selected_intent in {'creation', 'analysis'}:
            regenerated = self.executor.execute_task('creation', f'{cleaned}. incluye estructura y validación', context=context)
            if regenerated:
                response = clean_output(regenerated)
                verification = self.verifier.verify(cleaned, response, source_count=len(sources), executed=False)
        recent_failures = self.episode_learner.recent_failures(understanding.pattern_key, limit=3)
        failure_penalty = min(0.4, len(recent_failures) * 0.08 + verification.generic_penalty)
        confidence = self.confidence_engine.evaluate(
            intent_scores=understanding.intent_scores,
            selected_intent=understanding.selected_intent,
            verification_score=verification.score,
            memory_hit=memory_hit is not None,
            source_count=len(sources) or (1 if memory_hit else 0),
            route_confidence=decision.confidence_hint,
            failure_penalty=failure_penalty,
        )
        quality_score = round((verification.score * 0.55 + confidence.score * 0.45), 4)

        fallback_response = cognitive_result.response_text if cognitive_result.response_text and cognitive_result.response_text != response else None
        response = self.finalizer.finalize(cleaned, response, fallback=fallback_response)

        best_solution = {
            'task': understanding.selected_intent,
            'proposal': response,
            'final_score': quality_score,
            'verification_score': verification.score,
            'confidence': confidence.score,
            'route': route,
            'issues': verification.issues,
            'context_profile': context,
        }
        bucket = 'facts' if understanding.selected_intent in {'fact', 'time', 'date', 'identity', 'conversation'} else 'solutions'
        self.memory_store.put(bucket, cleaned, response, source=engine)
        self.memory_store.put('patterns', understanding.pattern_key, {
            'query': understanding.pattern_key,
            'value': response[:220],
            'source': route,
            'intent': understanding.selected_intent,
            'confidence': confidence.score,
        })
        if verification.issues:
            self.memory_store.put('failures', f'{cleaned}:{route}', {'query': cleaned, 'value': ';'.join(verification.issues), 'source': route})
        self.memory_store.put('episodes', f'{cleaned}:{quality_score}', {'query': cleaned, 'value': response[:220], 'source': route, 'score': quality_score})
        self.storage.save_learned_pattern(
            pattern_key=understanding.pattern_key,
            intent=understanding.selected_intent,
            route=route,
            confidence=confidence.score,
            support_count=len(self.storage.load_recent_episodes(pattern_key=understanding.pattern_key, limit=20)) + 1,
            sample_question=cleaned,
        )
        self.episode_learner.record_episode(
            question=question,
            normalized_question=cleaned,
            intent=understanding.selected_intent,
            route=route,
            tasks=understanding.tasks,
            response_text=response,
            confidence=confidence.score,
            verification_score=verification.score,
            quality_score=quality_score,
            sources=sources,
            issues=verification.issues,
            pattern_key=understanding.pattern_key,
            memory_hit=memory_hit is not None,
        )
        self.memory_agent.remember(cleaned, best_solution, understanding.selected_intent)
        self.session_context.register_turn(cleaned, response)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        for step in decision.steps:
            if step['step'] == 'execute':
                step['status'] = 'completed'
                step['detail'] = f"{engine}:{elapsed_ms:.1f}ms"
            elif step['step'] in {'verify', 'learn'}:
                step['status'] = 'completed'
        return AutonomousResult(
            question=cleaned,
            intent=understanding.selected_intent,
            tasks=understanding.tasks,
            plan=decision.steps,
            research={'used_learning': route in {'retrieve', 'analyze'}, 'sources': sources, 'requires_freshness': understanding.requires_freshness, 'confidence_band': confidence.band},
            best_solution=best_solution,
            memory=self.memory_store.export(),
            response_text=response,
        )

    def main_pipeline(self, question: str) -> str:
        """Compatibility wrapper exposing the full cognitive loop as plain text."""
        result = self.process(question)
        return result.response_text
