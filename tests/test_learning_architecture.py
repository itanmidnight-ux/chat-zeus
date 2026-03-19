from pathlib import Path

from src.autonomous_system import AutonomousReasoningSystem
from src.core.confidence import ConfidenceEngine
from src.core.decision import DecisionEngine
from src.core.understanding import SemanticUnderstandingEngine
from src.storage import StorageManager


def test_semantic_understanding_detects_creation_and_pattern() -> None:
    understanding = SemanticUnderstandingEngine().analyze('Diseña una arquitectura de IA con memoria limitada')
    assert understanding.selected_intent in {'creation', 'analysis'}
    assert understanding.pattern_key
    assert 'create' in understanding.route_candidates or 'analyze' in understanding.route_candidates


def test_decision_engine_prefers_math_route_for_symbolic_input() -> None:
    understanding = SemanticUnderstandingEngine().analyze('12 / 3 + 7')
    decision = DecisionEngine(max_memory_mb=256, max_external_queries=4).decide(understanding)
    assert decision.route == 'math'
    assert decision.engine == 'math'


def test_autonomous_system_persists_episode_and_strategy(tmp_path: Path) -> None:
    storage = StorageManager(tmp_path / 'knowledge.sqlite3', tmp_path / 'checkpoints')
    system = AutonomousReasoningSystem(storage=storage, memory_path=tmp_path / 'agent_memory.json')

    result = system.process('diseña una nave espacial segura')

    episodes = storage.load_recent_episodes(limit=5)
    assert result.best_solution['confidence'] > 0
    assert episodes
    assert episodes[0]['route'] in {'create', 'analyze', 'retrieve', 'math', 'execute', 'direct'}
    assert storage.load_strategy_stat(episodes[0]['route'], episodes[0]['pattern_key']) is not None


def test_confidence_engine_penalizes_repeated_failures() -> None:
    report = ConfidenceEngine().evaluate(
        intent_scores={'fact': 0.9},
        selected_intent='fact',
        verification_score=0.3,
        memory_hit=False,
        source_count=0,
        route_confidence=0.4,
        failure_penalty=0.5,
    )
    assert report.score < 0.55
    assert report.band in {'low', 'very_low'}
