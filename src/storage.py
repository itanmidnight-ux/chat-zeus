"""Persistencia en SQLite y checkpoints JSON para reanudación."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from src.utils import read_json, utc_now_iso, write_json


class StorageManager:
    def __init__(self, db_path: Path, checkpoint_dir: Path):
        self.db_path = db_path
        self.checkpoint_dir = checkpoint_dir
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                '''
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    response TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS simulation_runs (
                    run_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    state TEXT NOT NULL,
                    progress REAL NOT NULL,
                    result_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ml_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    features_json TEXT NOT NULL,
                    target REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                '''
            )
            seed_count = conn.execute('SELECT COUNT(*) FROM knowledge_documents').fetchone()[0]
            if seed_count == 0:
                docs = [
                    ('Ecuación de Tsiolkovski', 'rocketry', 'delta_v = ve * ln(m0/mf). Útil para estimar cambios de velocidad en vehículos cohete.', 'local_seed'),
                    ('Arrastre aerodinámico', 'physics', 'F_d = 0.5 * rho * Cd * A * v^2. Aproximación básica para atmósferas densas.', 'local_seed'),
                    ('Energía cinética', 'physics', 'E_k = 0.5 * m * v^2. Sirve para estimar energía asociada al movimiento.', 'local_seed'),
                    ('Gas ideal', 'thermo', 'P * V = n * R * T. Aproximación termodinámica básica para gases.', 'local_seed'),
                    ('Propulsión química', 'chemistry', 'La energía específica y la relación de mezcla afectan empuje, temperatura de cámara e impulso específico.', 'local_seed'),
                    ('Trayectoria balística simplificada', 'physics', 'El alcance depende de velocidad inicial, gravedad, drag y tiempo de combustión efectivo.', 'local_seed'),
                ]
                conn.executemany(
                    'INSERT INTO knowledge_documents(title, category, content, source, created_at) VALUES (?, ?, ?, ?, ?)',
                    [(title, category, content, source, utc_now_iso()) for title, category, content, source in docs],
                )

    def search_knowledge(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        keywords = [word.strip().lower() for word in query.split() if len(word.strip()) > 2][:8]
        if not keywords:
            return []
        clauses = ' OR '.join(['lower(title) LIKE ? OR lower(content) LIKE ? OR lower(category) LIKE ?' for _ in keywords])
        values: list[str] = []
        for keyword in keywords:
            like = f'%{keyword}%'
            values.extend([like, like, like])
        sql = f'SELECT title, category, content, source FROM knowledge_documents WHERE {clauses} LIMIT ?'
        with self._connect() as conn:
            rows = conn.execute(sql, (*values, limit)).fetchall()
        return [dict(row) for row in rows]

    def save_conversation(self, question: str, response: str, context_json: str) -> None:
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO conversations(question, response, context_json, created_at) VALUES (?, ?, ?, ?)',
                (question, response, context_json, utc_now_iso()),
            )

    def recent_conversations(self, limit: int = 5) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT question, response, context_json, created_at FROM conversations ORDER BY id DESC LIMIT ?',
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_run_state(self, run_id: str, question: str, state: str, progress: float, result_json: str) -> None:
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO simulation_runs(run_id, question, state, progress, result_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    state=excluded.state,
                    progress=excluded.progress,
                    result_json=excluded.result_json,
                    updated_at=excluded.updated_at
                ''',
                (run_id, question, state, progress, result_json, utc_now_iso()),
            )

    def load_run_state(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute('SELECT * FROM simulation_runs WHERE run_id = ?', (run_id,)).fetchone()
        return dict(row) if row else None

    def append_ml_observation(self, features_json: str, target: float) -> None:
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO ml_observations(features_json, target, created_at) VALUES (?, ?, ?)',
                (features_json, target, utc_now_iso()),
            )

    def load_ml_observations(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute('SELECT features_json, target FROM ml_observations ORDER BY id ASC').fetchall()
        return [dict(row) for row in rows]

    def checkpoint_path(self, run_id: str) -> Path:
        return self.checkpoint_dir / f'{run_id}.json'

    def load_checkpoint(self, run_id: str) -> dict[str, Any]:
        return read_json(self.checkpoint_path(run_id), default={})

    def save_checkpoint(self, run_id: str, payload: dict[str, Any]) -> None:
        write_json(self.checkpoint_path(run_id), payload)
