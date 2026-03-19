"""Agente de memoria persistente y aprendizaje incremental ligero."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.storage import StorageManager
from src.utils import read_json, write_json


class MemoryAgent:
    def __init__(self, storage: StorageManager, path: Path):
        self.storage = storage
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        return read_json(self.path, default={'success': [], 'failures': [], 'patterns': [], 'weights': {'feasibility': 0.4, 'efficiency': 0.25, 'safety': 0.35}})

    def remember(self, question: str, best_solution: dict[str, Any], intent: str) -> dict[str, Any]:
        memory = self.load()
        record = {
            'question': question,
            'task': best_solution.get('task', 'general'),
            'proposal': best_solution.get('proposal', ''),
            'score': float(best_solution.get('final_score', best_solution.get('score', 0.0))),
            'intent': intent,
        }
        bucket = 'success' if record['score'] >= 0.65 else 'failures'
        memory[bucket] = (memory.get(bucket, []) + [record])[-24:]
        pattern = {'task': record['task'], 'intent': intent, 'score': record['score']}
        memory['patterns'] = (memory.get('patterns', []) + [pattern])[-32:]
        weights = memory.get('weights', {'feasibility': 0.4, 'efficiency': 0.25, 'safety': 0.35})
        weights['feasibility'] = round(min(0.55, weights['feasibility'] + 0.005), 3)
        weights['safety'] = round(min(0.45, weights['safety'] + (0.004 if record['score'] < 0.7 else 0.002)), 3)
        weights['efficiency'] = round(max(0.15, 1.0 - weights['feasibility'] - weights['safety']), 3)
        memory['weights'] = weights
        write_json(self.path, memory)
        self.storage.save_model_state('autonomous_reasoner_memory', memory)
        self.storage.append_ml_observation(json.dumps({'intent': intent, 'task': record['task']}, ensure_ascii=False), record['score'], max(0.4, min(1.0, record['score'])))
        return memory
