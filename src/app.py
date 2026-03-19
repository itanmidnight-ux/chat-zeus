"""Punto de entrada de la aplicación CLI para Termux."""
from __future__ import annotations

from src.chatbot import ChatbotInterface
from src.config import CONFIG, ensure_directories
from src.external import ExternalKnowledgeFetcher
from src.knowledge import KnowledgeManager
from src.ml import LightweightMLModel
from src.optimizer import IterativeOptimizer
from src.reporting import ReportWriter
from src.simulation import SimulationEngine
from src.storage import StorageManager
from src.utils import setup_logging


def build_app() -> ChatbotInterface:
    ensure_directories()
    logger = setup_logging(CONFIG.logs_dir)
    storage = StorageManager(CONFIG.db_path, CONFIG.checkpoint_dir)
    knowledge = KnowledgeManager(storage)
    simulation = SimulationEngine(storage, chunk_size=CONFIG.simulation_chunk_size)
    ml_model = LightweightMLModel(storage)
    external_fetcher = ExternalKnowledgeFetcher(timeout_sec=CONFIG.internet_timeout_sec)
    optimizer = IterativeOptimizer(simulation)
    report_writer = ReportWriter(CONFIG.report_dir)
    return ChatbotInterface(storage, knowledge, simulation, ml_model, external_fetcher, optimizer, report_writer, logger)


def print_banner() -> None:
    print('Supercomputadora simplificada para Termux')
    print('Escribe una pregunta técnica o científica. Usa "salir" para terminar.')
    print('Puedes incluir parámetros como payload=120 fuel=240 thrust=18000 steps=500')


def main() -> None:
    app = build_app()
    print_banner()
    while True:
        question = input('\n> ').strip()
        if not question:
            print('Introduce una pregunta o escribe salir.')
            continue
        if question.lower() in {'salir', 'exit', 'quit'}:
            print('Sesión finalizada.')
            break
        result = app.safe_answer(question)
        print('\n' + result.get('response_text', 'Sin respuesta disponible.'))
        if result.get('report_path'):
            print(f"\n[reporte] Guardado en {result['report_path']}")
        if result.get('error'):
            print(f"[error] {result['error']}")
