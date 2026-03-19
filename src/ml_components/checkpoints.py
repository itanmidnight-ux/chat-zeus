"""Gestión de checkpoints JSON/SQLite del aprendizaje incremental."""
from __future__ import annotations

from typing import Any

from src.config import CONFIG
from src.utils import read_json, write_json


class CheckpointManager:
    def __init__(self, storage, checkpoint_name: str):
        self.storage = storage
        self.path = CONFIG.models_dir / checkpoint_name

    def load(self, model_name: str, default_state: dict[str, Any]) -> dict[str, Any]:
        checkpoint_state = self.storage.load_ml_checkpoint(model_name)
        if checkpoint_state:
            self.storage.save_model_state(model_name, checkpoint_state)
            write_json(self.path, checkpoint_state)
            return checkpoint_state
        state = self.storage.load_model_state(model_name)
        if state:
            write_json(self.path, state)
            return state
        json_state = read_json(self.path, {})
        if json_state:
            self.storage.save_model_state(model_name, json_state)
            return json_state
        self.save(model_name, default_state)
        return default_state

    def save(self, model_name: str, state: dict[str, Any]) -> None:
        self.storage.save_model_state(model_name, state)
        self.storage.save_ml_checkpoint(model_name, state)
        write_json(self.path, state)
