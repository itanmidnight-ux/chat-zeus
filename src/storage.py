"""Persistencia en SQLite y checkpoints JSON para reanudación."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.utils import read_json, utc_now_iso, write_json


SCHEMA_SQL = '''
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
CREATE TABLE IF NOT EXISTS research_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    findings_count INTEGER NOT NULL,
    quality_score REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    usefulness REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_state (
    name TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS connectivity_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
);
'''

SEED_DOCUMENTS = [
    ('Ecuación de Tsiolkovski', 'rocketry', 'delta_v = ve * ln(m0/mf). Útil para estimar cambios de velocidad en vehículos cohete.', 'local_seed'),
    ('Arrastre aerodinámico', 'physics', 'F_d = 0.5 * rho * Cd * A * v^2. Aproximación básica para atmósferas densas.', 'local_seed'),
    ('Energía cinética', 'physics', 'E_k = 0.5 * m * v^2. Sirve para estimar energía asociada al movimiento.', 'local_seed'),
    ('Gas ideal', 'thermo', 'P * V = n * R * T. Aproximación termodinámica básica para gases.', 'local_seed'),
    ('Propulsión química', 'chemistry', 'La energía específica y la relación de mezcla afectan empuje, temperatura de cámara e impulso específico.', 'local_seed'),
    ('Trayectoria balística simplificada', 'physics', 'El alcance depende de velocidad inicial, gravedad, drag y tiempo de combustión efectivo.', 'local_seed'),
    ('Análisis de sistemas complejos', 'systems', 'Dividir el problema en subsistemas, restricciones, riesgos, verificación y experimentos acelera la investigación seria.', 'local_seed'),
    ('Ingeniería de misión', 'mission', 'Un diseño de misión debe separar factibilidad física, coste energético, seguridad, materiales y operaciones.', 'local_seed'),
    ('Cálculo diferencial', 'mathematics', 'd/dx(x^n) = n*x^(n-1). Las derivadas ayudan a analizar tasas de cambio, optimización y sensibilidad.', 'local_seed'),
    ('Álgebra lineal', 'mathematics', 'det(A) permite evaluar invertibilidad; A^-1 existe si det(A) != 0. Las matrices modelan sistemas y transformaciones.', 'local_seed'),
    ('Litostática básica', 'geology', 'sigma = rho * g * h aproxima la presión litostática de una columna geológica.', 'local_seed'),
    ('Patente técnica - motor cohete', 'patent', 'Las patentes describen configuraciones, materiales, cámaras y toberas útiles para recuperar restricciones de diseño.', 'local_seed'),
    ('Resistencia de materiales', 'materials', 'esfuerzo = fuerza / area. El factor de seguridad compara resistencia del material frente al esfuerzo aplicado.', 'local_seed'),
]


class StorageManager:
    def __init__(self, db_path: Path, checkpoint_dir: Path):
        self.db_path = db_path
        self.checkpoint_dir = checkpoint_dir
        self._journal_mode = 'WAL'
        self._temp_store = 'FILE'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(f'PRAGMA journal_mode={self._journal_mode}')
        except sqlite3.OperationalError:
            self._journal_mode = 'DELETE'
            conn.execute(f'PRAGMA journal_mode={self._journal_mode}')
        conn.execute('PRAGMA synchronous=NORMAL')
        try:
            conn.execute(f'PRAGMA temp_store={self._temp_store}')
        except sqlite3.OperationalError:
            self._temp_store = 'MEMORY'
            conn.execute(f'PRAGMA temp_store={self._temp_store}')
        conn.execute('PRAGMA cache_size=-2048')
        return conn

    def _seed_documents(self, conn: sqlite3.Connection) -> None:
        seed_count = conn.execute('SELECT COUNT(*) FROM knowledge_documents').fetchone()[0]
        if seed_count:
            return
        conn.executemany(
            'INSERT INTO knowledge_documents(title, category, content, source, created_at) VALUES (?, ?, ?, ?, ?)',
            [(title, category, content, source, utc_now_iso()) for title, category, content, source in SEED_DOCUMENTS],
        )

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            self._seed_documents(conn)

    def _init_db(self) -> None:
        try:
            self._initialize_schema()
        except sqlite3.OperationalError as exc:
            if 'disk i/o error' not in str(exc).lower() or (self._journal_mode == 'DELETE' and self._temp_store == 'MEMORY'):
                raise
            self._journal_mode = 'DELETE'
            self._temp_store = 'MEMORY'
            self._initialize_schema()

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

    def recover_incomplete_runs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id, question, state, progress, result_json, updated_at FROM simulation_runs WHERE state != 'completed' ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def append_ml_observation(self, features_json: str, target: float) -> None:
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO ml_observations(features_json, target, created_at) VALUES (?, ?, ?)',
                (features_json, target, utc_now_iso()),
            )

    def load_ml_observations(self, limit: int | None = None) -> list[dict[str, Any]]:
        sql = 'SELECT features_json, target FROM ml_observations ORDER BY id DESC'
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += ' LIMIT ?'
            params = (max(1, int(limit)),)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        payload = [dict(row) for row in rows]
        payload.reverse()
        return payload

    def ml_observation_summary(self) -> dict[str, float]:
        with self._connect() as conn:
            row = conn.execute(
                '''
                SELECT COUNT(*) AS samples_seen,
                       AVG(target) AS avg_target,
                       MIN(target) AS min_target,
                       MAX(target) AS max_target
                FROM ml_observations
                '''
            ).fetchone()
        return {
            'samples_seen': float(row['samples_seen'] or 0.0),
            'avg_target': float(row['avg_target'] or 0.0),
            'min_target': float(row['min_target'] or 0.0),
            'max_target': float(row['max_target'] or 0.0),
        }

    def save_research_session(self, question: str, profile_json: str, findings_count: int, quality_score: float) -> None:
        created_at = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO research_sessions(question, profile_json, findings_count, quality_score, created_at) VALUES (?, ?, ?, ?, ?)',
                (question, profile_json, findings_count, quality_score, created_at),
            )
            profile = read_json_string(profile_json)
            for item in profile.get('findings', [])[:12]:
                usefulness = float(item.get('score', 0.0))
                conn.execute(
                    'INSERT INTO source_feedback(source_type, usefulness, created_at) VALUES (?, ?, ?)',
                    (item.get('source_type', 'unknown'), usefulness, created_at),
                )

    def load_recent_research_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT question, profile_json, findings_count, quality_score, created_at FROM research_sessions ORDER BY id DESC LIMIT ?',
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def source_performance_profile(self) -> dict[str, float]:
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT source_type, AVG(usefulness) AS avg_usefulness FROM source_feedback GROUP BY source_type'
            ).fetchall()
        return {str(row['source_type']): float(row['avg_usefulness']) for row in rows}

    def save_model_state(self, name: str, state: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO model_state(name, state_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET state_json=excluded.state_json, updated_at=excluded.updated_at
                ''',
                (name, json.dumps(state, ensure_ascii=False), utc_now_iso()),
            )

    def load_model_state(self, name: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute('SELECT state_json FROM model_state WHERE name = ?', (name,)).fetchone()
        if not row:
            return None
        return read_json_string(str(row['state_json']))

    def save_connectivity_event(self, source_type: str, status: str, latency_ms: float, detail: str) -> None:
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO connectivity_events(source_type, status, latency_ms, detail, created_at) VALUES (?, ?, ?, ?, ?)',
                (source_type, status, float(latency_ms), detail[:500], utc_now_iso()),
            )

    def connectivity_profile(self) -> dict[str, dict[str, float]]:
        with self._connect() as conn:
            rows = conn.execute(
                '''
                SELECT source_type,
                       AVG(CASE WHEN status = 'ok' THEN 1.0 ELSE 0.0 END) AS success_rate,
                       AVG(latency_ms) AS avg_latency_ms,
                       COUNT(*) AS total_events
                FROM connectivity_events
                GROUP BY source_type
                '''
            ).fetchall()
        return {
            str(row['source_type']): {
                'success_rate': float(row['success_rate'] or 0.0),
                'avg_latency_ms': float(row['avg_latency_ms'] or 0.0),
                'total_events': float(row['total_events'] or 0.0),
            }
            for row in rows
        }

    def checkpoint_path(self, run_id: str) -> Path:
        return self.checkpoint_dir / f'{run_id}.json'

    def load_checkpoint(self, run_id: str) -> dict[str, Any]:
        return read_json(self.checkpoint_path(run_id), default={})

    def save_checkpoint(self, run_id: str, payload: dict[str, Any]) -> None:
        write_json(self.checkpoint_path(run_id), payload)


def read_json_string(payload: str) -> dict[str, Any]:
    try:
        return json.loads(payload)
    except Exception:
        return {}
