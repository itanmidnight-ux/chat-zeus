from src.autonomous_system import AutonomousReasoningSystem
from src.core.intent import detect_intent_advanced
from src.engines.math_engine import solve_math
from src.sandbox.executor import execute_code_safely


def test_detect_intent_advanced_math():
    assert detect_intent_advanced('5 + 8') == 'math'


def test_solve_math_basic():
    assert solve_math('5 + 8') == '13'


def test_sandbox_safe_execution():
    assert execute_code_safely('result = sum(range(5)); print(result)') == '10'


def test_main_pipeline_formula_response(tmp_path):
    agent = AutonomousReasoningSystem(memory_path=tmp_path / 'agent_memory.json')
    assert 'cateto' in agent.main_pipeline('formula para cateto').lower()
