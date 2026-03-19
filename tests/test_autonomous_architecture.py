import tempfile
import unittest
from pathlib import Path

from src.agents.memory import MemoryAgent
from src.autonomous_system import AutonomousReasoningSystem
from src.core.decomposer import decompose_problem
from src.core.planner import build_research_plan
from src.handlers import handle_simple_queries
from src.intent import classify_intent
from src.storage import StorageManager


class AutonomousArchitectureTests(unittest.TestCase):
    def test_reality_handler_returns_time_or_date_response(self) -> None:
        response = handle_simple_queries('qué hora es')
        self.assertIsNotNone(response)
        self.assertIn('Son las', response)

    def test_intent_and_decomposer_cover_requested_layers(self) -> None:
        self.assertEqual(classify_intent('qué hora es'), 'simple')
        self.assertEqual(classify_intent('cómo funciona un cohete'), 'explicativa')
        self.assertEqual(classify_intent('diseña una nave espacial'), 'analitica')
        tasks = decompose_problem('diseña una nave espacial')
        self.assertIn('propulsión', tasks)
        plan = build_research_plan('diseña una nave espacial segura', tasks)
        self.assertGreaterEqual(plan[0]['priority'], plan[-1]['priority'])

    def test_multi_agent_loop_generates_filtered_response_and_persists_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = StorageManager(Path(tmp) / 'knowledge.sqlite3', Path(tmp) / 'checkpoints')
            memory_agent = MemoryAgent(storage, Path(tmp) / 'memory.json')
            system = AutonomousReasoningSystem(storage, memory_agent)

            result = system.process('diseña una nave espacial para investigación segura')

            self.assertTrue(result.response_text)
            self.assertNotIn('simulación', result.response_text.lower())
            self.assertTrue((Path(tmp) / 'memory.json').exists())
            memory = memory_agent.load()
            self.assertGreaterEqual(len(memory['patterns']), 1)


if __name__ == '__main__':
    unittest.main()
