"""Aprendizaje online y reentrenamiento en bloques pequeños."""
from __future__ import annotations

import gc
import json
from typing import Any

from src.config import CONFIG


class IncrementalLearner:
    def __init__(self, storage, feature_names: list[str]):
        self.storage = storage
        self.feature_names = feature_names
        self.max_batch = max(4, min(16, CONFIG.storage_stream_batch_size // 4))

    def update_state(self, state: dict[str, Any], features: dict[str, float], target: float, reliability: float) -> dict[str, Any]:
        samples_seen = int(state.get('samples_seen', 0)) + 1
        learning_rate = float(state.get('learning_rate', 0.0125)) * max(0.15, reliability)
        for name in self.feature_names:
            old_mean = float(state['feature_mean'].get(name, 0.0))
            delta = features.get(name, 0.0) - old_mean
            new_mean = old_mean + delta / samples_seen
            state['feature_mean'][name] = new_mean
            state['feature_scale'][name] = max(1.0, float(state['feature_scale'].get(name, 1.0)) * 0.985 + abs(delta) * 0.015)

        normalized = {
            name: (features.get(name, 0.0) - float(state['feature_mean'].get(name, 0.0))) / max(float(state['feature_scale'].get(name, 1.0)), 1.0)
            for name in self.feature_names
        }
        prediction = float(state.get('bias', 0.0))
        for name in self.feature_names:
            prediction += float(state['weights'].get(name, 0.0)) * normalized[name]
        error = target - prediction
        state['bias'] = float(state.get('bias', 0.0)) + learning_rate * error * 0.04
        for name in self.feature_names:
            weight = float(state['weights'].get(name, 0.0))
            state['weights'][name] = weight + learning_rate * error * normalized[name] * 0.01
        previous_loss = float(state.get('loss_ema', 0.0))
        state['loss_ema'] = previous_loss * 0.9 + abs(error) * 0.1
        state['samples_seen'] = samples_seen
        state['last_reliability'] = reliability
        return state

    def retrain_in_background(self, state: dict[str, Any]) -> dict[str, Any]:
        observations = self.storage.load_ml_observations(limit=self.max_batch)
        for item in observations:
            try:
                features = json.loads(item.get('features_json', '{}'))
            except json.JSONDecodeError:
                continue
            state = self.update_state(state, features, float(item.get('target', 0.0)), float(item.get('reliability', 0.35)))
            gc.collect()
        return state
