"""Memory agent with lightweight incremental learning and pattern tracking."""
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
        return read_json(self.path, default={'success': [], 'failures': [], 'patterns': [], 'weights': {'feasibility': 0.4, 'efficiency': 0.3, 'safety': 0.3}})

    def update_memory(self, result: dict[str, Any]) -> dict[str, Any]:
        try:
            memory = self.load()
            score = float(result.get('final_score', result.get('score', 0.0)))
            bucket = 'success' if score >= 0.65 else 'failures'
            compact = {
                'question': result.get('question', ''),
                'task': result.get('task', 'general'),
                'proposal': result.get('proposal', ''),
                'score': score,
                'intent': result.get('intent', 'analytical'),
            }
            memory[bucket] = (memory.get(bucket, []) + [compact])[-24:]
            memory['patterns'] = (memory.get('patterns', []) + [{'task': compact['task'], 'intent': compact['intent'], 'score': compact['score']}])[-32:]
            weights = memory.get('weights', {'feasibility': 0.4, 'efficiency': 0.3, 'safety': 0.3})
            weights['feasibility'] = round(min(0.55, weights['feasibility'] + 0.005), 3)
            weights['safety'] = round(min(0.45, weights['safety'] + (0.004 if score < 0.7 else 0.002)), 3)
            weights['efficiency'] = round(max(0.15, 1.0 - weights['feasibility'] - weights['safety']), 3)
            memory['weights'] = weights
            write_json(self.path, memory)
            self.storage.save_model_state('autonomous_reasoner_memory', memory)
            self.storage.append_ml_observation(json.dumps({'intent': compact['intent'], 'task': compact['task']}, ensure_ascii=False), compact['score'], max(0.4, min(1.0, compact['score'])))
            return memory
        except Exception:
            return self.load()

    def remember(self, question: str, best_solution: dict[str, Any], intent: str) -> dict[str, Any]:
        return self.update_memory({**best_solution, 'question': question, 'intent': intent})
