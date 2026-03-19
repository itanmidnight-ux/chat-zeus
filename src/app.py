"""Punto de entrada de la aplicación CLI para Termux."""
from __future__ import annotations

from src.chatbot import ChatbotInterface
from src.calculator import AnalyticalCalculator
from src.config import CONFIG, ensure_directories
from src.external import ExternalKnowledgeFetcher
from src.knowledge import KnowledgeManager
from src.ml import LightweightMLModel
from src.optimizer import IterativeOptimizer
from src.reporting import ReportWriter
from src.simulation import SimulationEngine
from src.storage import StorageManager
from src.termux_ui import TermuxUI
from src.utils import apply_soft_memory_limit, ensure_environment_defaults, setup_logging
from src.worker import BackgroundExecutor


def build_app() -> ChatbotInterface:
    ensure_environment_defaults()
    apply_soft_memory_limit(CONFIG.max_task_memory_mb)
    ensure_directories()
    logger = setup_logging(CONFIG.logs_dir)
    storage = StorageManager(CONFIG.db_path, CONFIG.checkpoint_dir)
    knowledge = KnowledgeManager(storage)
    simulation = SimulationEngine(storage, chunk_size=CONFIG.simulation_chunk_size)
    calculator = AnalyticalCalculator()
    ml_model = LightweightMLModel(storage)
    external_fetcher = ExternalKnowledgeFetcher(storage, timeout_sec=CONFIG.internet_timeout_sec, max_queries=CONFIG.max_external_queries, max_retries=CONFIG.internet_max_retries)
    optimizer = IterativeOptimizer(simulation, storage)
    report_writer = ReportWriter(CONFIG.report_dir)
    background_executor = BackgroundExecutor(max_workers=CONFIG.max_workers)
    return ChatbotInterface(storage, knowledge, simulation, calculator, ml_model, external_fetcher, optimizer, report_writer, background_executor, logger)


def main() -> None:
    app = build_app()
    ui = TermuxUI()
    print(ui.render_welcome())
    try:
        while True:
            try:
                question = input(ui.prompt()).strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                break
            if not question:
                continue
            if question.lower() in {'salir', 'exit', 'quit'}:
                break
            result = app.safe_answer(question)
            print(ui.render_response(result.get('response_text', 'Sin respuesta disponible.')))
    finally:
        app.background_executor.shutdown()
