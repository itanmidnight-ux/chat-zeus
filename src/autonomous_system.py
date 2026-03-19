"""Autonomous reasoning system implementing a low-memory learning pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from src.agents.memory import MemoryAgent
from src.config import CONFIG
from src.core.confidence import ConfidenceEngine
from src.core.decision import DecisionEngine
from src.core.episodic import EpisodeLearner
from src.core.executor import TaskExecutor
from src.core.learning import LearningEngine
from src.core.memory import LightweightMemory
from src.core.understanding import SemanticUnderstandingEngine
from src.core.verifier import VerificationEngine
from src.engines.fact_engine import FactEngine
from src.storage import StorageManager
from src.utils.filters import clean_input, clean_output
from src.utils.handlers import handle_simple_queries


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

    def _memory_lookup(self, understanding) -> dict[str, Any] | None:
        for bucket in ('facts', 'solutions', 'patterns'):
            hit = self.memory_store.get(bucket, understanding.normalized_question)
            if hit:
                return {**hit, 'bucket': bucket}
        learned = self.storage.load_learned_pattern(understanding.pattern_key)
        if learned:
            return {'bucket': 'patterns', 'value': learned.get('sample_question', ''), 'source': 'strategy'}
        return None

    def _execute_plan(self, engine: str, question: str) -> tuple[str, list[str]]:
        sources: list[str] = []
        if engine == 'analysis':
            local_hits = self.storage.search_knowledge(question, limit=3)
            if local_hits:
                summary = ' '.join(item['content'] for item in local_hits[:2])
                sources.extend([str(item['source']) for item in local_hits[:2]])
                response = self.executor.execute_task('analysis', f'{question}. contexto: {summary}')
                return response, sources
        response = self.executor.execute_task(engine, question)
        if engine == 'fact':
            cached = self.memory_store.get('facts', question)
            if cached:
                sources.append(str(cached.get('source', 'memory')))
        return response, sources

    def main_pipeline(self, question: str) -> str:
        return self.process(question).response_text

    def process(self, question: str) -> AutonomousResult:
        started = time.perf_counter()
        cleaned = clean_input(question)
        direct = handle_simple_queries(cleaned)
        if direct:
            response = clean_output(direct)
            verification = self.verifier.verify(cleaned, response, source_count=1)
            confidence = self.confidence_engine.evaluate(
                intent_scores={'simple': 0.95},
                selected_intent='simple',
                verification_score=verification.score,
                memory_hit=False,
                source_count=1,
                route_confidence=0.95,
                failure_penalty=0.0,
            )
            best_solution = {'task': 'simple', 'proposal': response, 'final_score': verification.score, 'confidence': confidence.score}
            self.memory_agent.remember(cleaned, best_solution, 'simple')
            return AutonomousResult(
                question=cleaned,
                intent='simple',
                tasks=['simple'],
                plan=[{'step': 'direct_response', 'status': 'completed'}],
                research={'used_learning': False, 'sources': ['clock']},
                best_solution=best_solution,
                memory=self.memory_store.export(),
                response_text=response,
            )

        understanding = self.understanding.analyze(cleaned)
        memory_hit = self._memory_lookup(understanding)
        decision = self.decision_engine.decide(understanding, hot_memory=memory_hit)
        route = decision.route
        engine = decision.engine
        response, sources = self._execute_plan(engine, cleaned)
        if not response and route != 'retrieve':
            response = self.learning_engine.search_and_learn(cleaned)
            sources.append('internet')
        response = clean_output(response or 'No pude completar la solicitud con suficiente confianza.')
        verification = self.verifier.verify(cleaned, response, source_count=len(sources), executed=engine == 'execution')
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

        if confidence.band in {'low', 'very_low'} and 'No pude completar' not in response:
            response = f'{response} Confianza estimada: {confidence.band}; conviene verificar los puntos clave.'

        best_solution = {
            'task': understanding.selected_intent,
            'proposal': response,
            'final_score': quality_score,
            'verification_score': verification.score,
            'confidence': confidence.score,
            'route': route,
            'issues': verification.issues,
        }
        bucket = 'facts' if understanding.selected_intent in {'fact', 'time', 'date'} else 'solutions'
        self.memory_store.put(bucket, cleaned, response, source=engine)
        self.memory_store.put('patterns', understanding.pattern_key, {
            'query': understanding.pattern_key,
            'value': response[:220],
            'source': route,
            'intent': understanding.selected_intent,
            'confidence': confidence.score,
        })
        if verification.issues:
            self.memory_store.put('failures', f'{cleaned}:{route}', {
                'query': cleaned,
                'value': ';'.join(verification.issues),
                'source': route,
            })
        self.memory_store.put('episodes', f'{cleaned}:{quality_score}', {
            'query': cleaned,
            'value': response[:220],
            'source': route,
            'score': quality_score,
        })
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
            research={
                'used_learning': route in {'retrieve', 'analyze'},
                'sources': sources,
                'requires_freshness': understanding.requires_freshness,
                'confidence_band': confidence.band,
            },
            best_solution=best_solution,
            memory=self.memory_store.export(),
            response_text=response,
        )
