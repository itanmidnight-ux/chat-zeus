"""Interfaz conversacional principal que coordina conocimiento, simulación, ML y reportes."""
from __future__ import annotations

import json
import re
from typing import Any

from src.external import ExternalKnowledgeFetcher
from src.knowledge import KnowledgeManager
from src.ml import LightweightMLModel
from src.optimizer import IterativeOptimizer
from src.reporting import ReportWriter
from src.simulation import SimulationEngine
from src.storage import StorageManager
from src.utils import safe_error_message


class ChatbotInterface:
    def __init__(
        self,
        storage: StorageManager,
        knowledge: KnowledgeManager,
        simulation: SimulationEngine,
        ml_model: LightweightMLModel,
        external_fetcher: ExternalKnowledgeFetcher,
        optimizer: IterativeOptimizer,
        report_writer: ReportWriter,
        logger,
    ):
        self.storage = storage
        self.knowledge = knowledge
        self.simulation = simulation
        self.ml_model = ml_model
        self.external_fetcher = external_fetcher
        self.optimizer = optimizer
        self.report_writer = report_writer
        self.logger = logger

    def _extract_defaults(self, question: str) -> dict[str, float | int]:
        defaults: dict[str, float | int] = {}
        patterns = {
            'payload_mass_kg': r'payload\s*=\s*([0-9]+(?:\.[0-9]+)?)',
            'fuel_mass_kg': r'fuel\s*=\s*([0-9]+(?:\.[0-9]+)?)',
            'dry_mass_kg': r'dry\s*=\s*([0-9]+(?:\.[0-9]+)?)',
            'thrust_n': r'thrust\s*=\s*([0-9]+(?:\.[0-9]+)?)',
            'steps': r'steps\s*=\s*([0-9]+)',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                defaults[key] = int(value) if key == 'steps' else value
        return defaults

    def _progress(self, run_id: str, progress: float) -> None:
        self.logger.info('Progreso %s: %.1f%%', run_id, progress * 100)
        print(f'[progreso] {run_id}: {progress * 100:.1f}%')

    def answer(self, question: str) -> dict[str, Any]:
        knowledge = self.knowledge.retrieve(question)
        defaults = self._extract_defaults(question)
        simulation_request = self.simulation.build_request(question, defaults)
        simulation = self.simulation.run(simulation_request, progress_callback=self._progress)
        self.ml_model.train_from_result(simulation)
        ml_result = self.ml_model.predict(simulation)
        external = self.external_fetcher.fetch_formula_hint(question)

        optimization = None
        if any(word in question.lower() for word in ['optimiza', 'optimize', 'mejora', 'improve']):
            optimization = self.optimizer.optimize(question, progress_callback=self._progress)

        analysis = (
            'Se combinó conocimiento local, una simulación de ascenso vertical simplificada, '
            'estimaciones de arrastre y una heurística de aprendizaje incremental. '
            f"Resumen RAG: {knowledge.summary}"
        )
        payload = {
            'analysis': analysis,
            'knowledge': knowledge.summary,
            'simulation': simulation,
            'ml': {
                'prediction': ml_result.prediction,
                'confidence': ml_result.confidence,
                'hypotheses': ml_result.hypotheses,
            },
            'external': external,
            'optimization': optimization,
            'recent_context': self.storage.recent_conversations(limit=3),
        }
        report_path = self.report_writer.save(question, payload)
        response_text = self.report_writer.render_text(payload)
        self.storage.save_conversation(question, response_text, json.dumps(payload, ensure_ascii=False))
        payload['report_path'] = str(report_path)
        payload['response_text'] = response_text
        return payload

    def safe_answer(self, question: str) -> dict[str, Any]:
        try:
            return self.answer(question)
        except MemoryError as exc:
            message = 'Se detectó un fallo de memoria. Reduce el tamaño del problema o los pasos de simulación.'
            self.logger.error(message)
            return {'response_text': message, 'error': safe_error_message(exc)}
        except ZeroDivisionError as exc:
            message = 'Se evitó una división por cero. Revisa parámetros de masa, empuje o velocidad de escape.'
            self.logger.error(message)
            return {'response_text': message, 'error': safe_error_message(exc)}
        except Exception as exc:
            self.logger.exception('Error inesperado al responder')
            return {
                'response_text': 'Ocurrió un error recuperable; consulta el log y reintenta con un problema más acotado.',
                'error': safe_error_message(exc),
            }
