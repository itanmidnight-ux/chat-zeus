"""Ejecución silenciosa en segundo plano con límite adaptable para Linux."""
from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


class BackgroundExecutor:
    def __init__(self, max_workers: int = 2):
        self.max_workers = max(1, max_workers)
        self._semaphore = threading.BoundedSemaphore(value=self.max_workers)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix='chat-zeus')

    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> Future:
        if not self._semaphore.acquire(blocking=False):
            return self._run_inline(fn, *args, **kwargs)
        try:
            future = self.executor.submit(fn, *args, **kwargs)
            future.add_done_callback(lambda _: self._semaphore.release())
            return future
        except RuntimeError:
            self._semaphore.release()
            return self._run_inline(fn, *args, **kwargs)
        except OSError as exc:
            self._semaphore.release()
            if 'cannot allocate memory' in str(exc).lower():
                return self._run_inline(fn, *args, **kwargs)
            raise

    def map_inline_safe(self, callables: list[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]]) -> list[Any]:
        results: list[Any] = []
        for fn, args, kwargs in callables:
            results.append(self.submit(fn, *args, **kwargs).result())
        return results

    def _run_inline(self, fn: Callable[..., Any], *args, **kwargs) -> Future:
        fallback_future: Future = Future()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            fallback_future.set_exception(exc)
        else:
            fallback_future.set_result(result)
        return fallback_future

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)
