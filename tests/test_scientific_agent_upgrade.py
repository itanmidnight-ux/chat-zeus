import re
from pathlib import Path

from src.autonomous_system import AutonomousReasoningSystem
from src.engines.creation_engine import CreationEngine
from src.storage import StorageManager


def test_simple_router_uses_system_clock_for_time(tmp_path: Path) -> None:
    system = AutonomousReasoningSystem(memory_path=tmp_path / 'agent_memory.json')
    result = system.process('qué hora es')

    assert result.intent == 'time'
    assert result.plan == [{'step': 'route_simple_intent', 'status': 'completed'}]
    assert re.search(r'\d{2}:\d{2}:\d{2}', result.response_text)


def test_memory_priority_answers_name_without_external_lookup(tmp_path: Path) -> None:
    storage = StorageManager(tmp_path / 'knowledge.sqlite3', tmp_path / 'checkpoints')
    system = AutonomousReasoningSystem(storage=storage, memory_path=tmp_path / 'agent_memory.json')

    system.process('me llamo Carla')
    result = system.process('como me llamo')

    assert result.intent == 'identity'
    assert result.response_text == 'Te llamas Carla.'
    assert result.best_solution['route'] == 'simple_router'


def test_finalizer_removes_confidence_noise(tmp_path: Path) -> None:
    system = AutonomousReasoningSystem(memory_path=tmp_path / 'agent_memory.json')

    cleaned = system.finalizer.finalize('qué hora es', 'Son las 10:00. Confianza estimada: low.')

    assert 'Confianza estimada' not in cleaned
    assert cleaned == 'Son las 10:00.'


def test_creation_engine_returns_scientific_structure() -> None:
    response = CreationEngine().build_solution('diseña una nave espacial con baja masa y control estable')

    assert 'Resultado final:' in response
    assert 'Hipótesis viables:' in response
    assert 'Factibilidad y validación:' in response
    assert 'Mejora futura:' in response
