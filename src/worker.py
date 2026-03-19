"""Ejecución silenciosa en segundo plano con un pool mínimo para Termux."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


class BackgroundExecutor:
    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix='chat-zeus')

    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> Future:
        return self.executor.submit(fn, *args, **kwargs)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)
