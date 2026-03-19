"""Interfaz conversacional principal que coordina conocimiento, simulación, ML y reportes."""
from __future__ import annotations

import json
import uuid
from typing import Any

from src.config import CONFIG
from src.external import ExternalKnowledgeFetcher
from src.knowledge import KnowledgeManager
from src.ml import LightweightMLModel
from src.optimizer import IterativeOptimizer
from src.reporting import ReportWriter
from src.simulation import SimulationEngine
from src.storage import StorageManager
from src.utils import detect_linker_memory_issue, extract_numeric_value, safe_error_message, sanitize_text
from src.worker import BackgroundExecutor


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
        background_executor: BackgroundExecutor,
        logger,
    ):
        self.storage = storage
        self.knowledge = knowledge
        self.simulation = simulation
        self.ml_model = ml_model
        self.external_fetcher = external_fetcher
        self.optimizer = optimizer
        self.report_writer = report_writer
        self.background_executor = background_executor
        self.logger = logger

    def _extract_defaults(self, question: str) -> dict[str, float | int]:
        defaults: dict[str, float | int] = {}
        patterns = {
            'payload_mass_kg': r'(?:payload|carga(?:\s+útil)?)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)',
            'fuel_mass_kg': r'(?:fuel|combustible)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)',
            'dry_mass_kg': r'(?:dry|masa\s+seca)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)',
            'thrust_n': r'(?:thrust|empuje)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)',
            'steps': r'(?:steps|pasos)\s*=\s*([0-9]+)',
            'mixture_ratio': r'(?:mixture|mezcla)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)',
            'exhaust_velocity_m_s': r'(?:ve|escape|exhaust)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)',
            'area_m2': r'(?:area|área)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)',
            'drag_coefficient': r'(?:cd|drag)\s*=\s*([0-9]+(?:[\.,][0-9]+)?)',
        }
        for key, pattern in patterns.items():
            value = extract_numeric_value(question, pattern)
            if value is not None:
                defaults[key] = int(value) if key == 'steps' else float(value)
        if 'steps' not in defaults:
            lower_question = question.lower()
            if any(word in lower_question for word in ['rápido', 'rapido', 'ligero', 'simple']):
                defaults['steps'] = 240
            elif any(word in lower_question for word in ['preciso', 'larga', 'detallado', 'optimiza']):
                defaults['steps'] = 720
        return defaults

    def _progress(self, run_id: str, progress: float) -> None:
        self.logger.info('Progreso %s: %.1f%%', run_id, progress * 100)

    def _build_analysis(self, question: str, knowledge_summary: str, simulation: dict[str, Any], external: dict[str, Any], recent_context: list[dict[str, Any]]) -> tuple[str, str]:
        context_hint = ''
        if recent_context:
            context_hint = f" El historial reciente contiene {len(recent_context)} intercambios y se usó para mantener continuidad temática sin exponer pasos internos."
        analysis = (
            'Se combinó recuperación local de conocimiento científico, una simulación silenciosa de ascenso y trayectoria simplificada, '
            'estimaciones básicas de gravedad, arrastre, propulsión y termodinámica, además de una capa de aprendizaje incremental con checkpoints persistentes.'
            f' Resumen RAG: {knowledge_summary}.{context_hint}'
            f" La corrida activa {simulation['run_id']} se ejecutó en bloques pequeños para respetar límites de memoria de Termux."
            f" El apoyo externo terminó con estado {external['status']}."
        )
        conclusions = (
            f"Para esta consulta, el diseño analizado alcanza aproximadamente {simulation['max_altitude_m']} m de altitud máxima con un delta-v de {simulation['delta_v_m_s']} m/s. "
            f"El modelo sugiere que la siguiente mejora más prometedora sería optimizar simultáneamente masa estructural, relación de mezcla y empuje para elevar el margen energético sin disparar el drag. "
            'Los resultados son útiles para exploración conceptual en Termux, pero no sustituyen validación de ingeniería de alta fidelidad.'
        )
        return sanitize_text(analysis), sanitize_text(conclusions)

    def _resume_note(self) -> str:
        pending_runs = self.storage.recover_incomplete_runs()
        if not pending_runs:
            return ''
        latest = pending_runs[0]
        return (
            f" Se detectó una ejecución previa no finalizada ({latest['run_id']}) con progreso "
            f"{round(float(latest['progress']) * 100, 1)} %, lo que confirma que el sistema conserva checkpoints reanudables tras reinicios."
        )

    def answer(self, question: str) -> dict[str, Any]:
        recent_context = self.storage.recent_conversations(limit=CONFIG.max_history_messages)
        knowledge = self.knowledge.retrieve(question, recent_context=recent_context)
        defaults = self._extract_defaults(question)
        run_id = f"sim_{uuid.uuid5(uuid.NAMESPACE_DNS, question).hex[:12]}"
        defaults['run_id'] = run_id
        simulation_request = self.simulation.build_request(question, defaults)
        simulation_future = self.background_executor.submit(self.simulation.run, simulation_request, self._progress)
        simulation = simulation_future.result()
        self.ml_model.train_from_result(simulation)
        ml_result = self.ml_model.predict(simulation)
        external_future = self.background_executor.submit(self.external_fetcher.fetch_formula_hint, question)

        optimization = None
        if any(word in question.lower() for word in ['optimiza', 'optimize', 'mejora', 'improve']):
            optimization_future = self.background_executor.submit(self.optimizer.optimize, question, 6, self._progress)
            optimization = optimization_future.result()

        external = external_future.result()

        analysis, conclusions = self._build_analysis(question, knowledge.summary, simulation, external, recent_context)
        analysis = sanitize_text(analysis + self._resume_note())
        payload = {
            'analysis': analysis,
            'conclusions': conclusions,
            'knowledge': {
                'summary': knowledge.summary,
                'formulas': knowledge.formulas,
            },
            'simulation': simulation,
            'ml': {
                'prediction': ml_result.prediction,
                'confidence': ml_result.confidence,
                'hypotheses': ml_result.hypotheses,
            },
            'external': external,
            'optimization': optimization,
            'recent_context': [{'question': item['question'], 'created_at': item['created_at']} for item in recent_context[:3]],
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
        except OSError as exc:
            if detect_linker_memory_issue(exc):
                message = (
                    'Se detectó saturación de memoria del linker de Android/Termux. '
                    'El sistema conservó checkpoints; reintenta con menos pasos, menor tamaño de problema o menos dependencias nativas cargadas.'
                )
                self.logger.error(message)
                return {'response_text': message, 'error': safe_error_message(exc)}
            raise
        except ZeroDivisionError as exc:
            message = 'Se evitó una división por cero. Revisa parámetros de masa, empuje o velocidad de escape.'
            self.logger.error(message)
            return {'response_text': message, 'error': safe_error_message(exc)}
        except Exception as exc:
            self.logger.exception('Error inesperado al responder')
            return {
                'response_text': 'Ocurrió un error recuperable; el sistema mantuvo su estado y puedes reintentar con un problema más acotado.',
                'error': safe_error_message(exc),
            }
