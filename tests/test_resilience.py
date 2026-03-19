import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.app import build_app
from src.config import CONFIG
from src.simulation import SimulationEngine
from src.storage import StorageManager
from src.utils import estimate_step_budget


class ResourceBudgetTests(unittest.TestCase):
    def test_step_budget_is_trimmed_to_safe_cap(self) -> None:
        budget = estimate_step_budget(requested_steps=2000, chunk_size=24, memory_mb=384, max_cap=480)
        self.assertLessEqual(budget, 480)
        self.assertGreaterEqual(budget, 24)

    def test_simulation_request_tracks_requested_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = StorageManager(Path(tmp) / 'knowledge.sqlite3', Path(tmp) / 'checkpoints')
            engine = SimulationEngine(storage, chunk_size=24)
            request = engine.build_request('cohete de prueba', {'steps': 2000})
            self.assertEqual(request.requested_steps, 2000)
            self.assertLessEqual(request.steps, CONFIG.hard_step_cap)


class MLBackendSafetyTests(unittest.TestCase):
    def test_uses_heuristic_backend_by_default_without_native_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch('importlib.metadata.distribution', side_effect=AssertionError('should not probe native backends')), patch.dict(os.environ, {}, clear=False):
            from src.ml import LightweightMLModel

            storage = StorageManager(Path(tmp) / 'knowledge.sqlite3', Path(tmp) / 'checkpoints')
            model = LightweightMLModel(storage)

            self.assertEqual(model.backend, 'dedicated-online-heuristic')

    def test_env_override_can_force_safe_backend_without_probings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'CHAT_ZEUS_ML_BACKEND': 'tflite_runtime'}, clear=False):
            from src.ml import LightweightMLModel

            storage = StorageManager(Path(tmp) / 'knowledge.sqlite3', Path(tmp) / 'checkpoints')
            model = LightweightMLModel(storage)

            self.assertEqual(model.backend, 'tflite_runtime')


class TermuxUITests(unittest.TestCase):
    def test_render_welcome_mentions_prompt_readiness(self) -> None:
        from src.termux_ui import TermuxUI

        ui = TermuxUI(colors=False)
        welcome = ui.render_welcome()

        self.assertIn('Haz tu pregunta cuando quieras', welcome)
        self.assertIn('Pregunta >', ui.prompt())


class ChatbotFallbackTests(unittest.TestCase):
    def test_safe_answer_returns_text_fallback_after_memory_error(self) -> None:
        app = build_app()
        try:
            original_answer = app.answer

            def raise_memory_error(question: str):
                raise MemoryError('simulated oom')

            app.answer = raise_memory_error  # type: ignore[assignment]
            result = app.safe_answer('Explica la resistencia de materiales con carga=10 area=2')
            self.assertIn('Análisis completo del problema:', result['response_text'])
        finally:
            app.answer = original_answer  # type: ignore[assignment]
            app.background_executor.shutdown()


if __name__ == '__main__':
    unittest.main()
