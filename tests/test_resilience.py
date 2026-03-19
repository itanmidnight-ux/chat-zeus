import json
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
            self.assertGreaterEqual(request.chunk_size or 0, 24)


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

    def test_model_state_is_persisted_to_json_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from src.ml import LightweightMLModel

            storage = StorageManager(Path(tmp) / 'knowledge.sqlite3', Path(tmp) / 'checkpoints')
            model = LightweightMLModel(storage)
            model.checkpoints.path = Path(tmp) / CONFIG.ml_checkpoint_file
            model.model_state_path = model.checkpoints.path
            model._persist_state()
            model.train_from_result({
                'delta_v_m_s': 100.0,
                'max_altitude_m': 50.0,
                'burn_time_s': 3.0,
                'payload_mass_kg': 10.0,
                'range_m': 20.0,
                'chemistry': {'estimated_efficiency': 0.5},
            })
            state_path = Path(tmp) / CONFIG.ml_checkpoint_file
            self.assertTrue(state_path.exists())
            payload = json.loads(state_path.read_text())
            self.assertGreaterEqual(payload['samples_seen'], 1)

    def test_prediction_exposes_variables_uncertainty_and_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from src.ml import LightweightMLModel

            storage = StorageManager(Path(tmp) / 'knowledge.sqlite3', Path(tmp) / 'checkpoints')
            model = LightweightMLModel(storage)
            simulation = {
                'delta_v_m_s': 120.0,
                'max_altitude_m': 80.0,
                'burn_time_s': 4.0,
                'payload_mass_kg': 12.0,
                'range_m': 30.0,
                'drag_coefficient': 0.7,
                'chemistry': {'estimated_efficiency': 0.45, 'mixture_ratio': 2.2},
            }
            model.train_from_result(simulation, question='analiza integral y matriz de un cohete', knowledge_summary='delta_v y arrastre')
            result = model.predict(simulation, question='analiza integral y matriz de un cohete', knowledge_summary='delta_v y arrastre')

            self.assertGreater(len(result.variables_considered), 0)
            self.assertGreater(len(result.uncertainty_drivers), 0)
            self.assertGreater(len(result.recommendations), 0)


class TermuxUITests(unittest.TestCase):
    def test_render_welcome_mentions_prompt_readiness(self) -> None:
        from src.termux_ui import TermuxUI

        ui = TermuxUI(colors=False)
        welcome = ui.render_welcome()

        self.assertIn('Listo para preguntas en lenguaje natural', welcome)
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
