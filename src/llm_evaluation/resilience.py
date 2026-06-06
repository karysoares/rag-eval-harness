"""Utilitários de resiliência para produção: retry e lock de ficheiro."""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    backoff_seconds: tuple[float, ...] = (0.2, 0.6, 1.2),
    jitter_ratio: float = 0.15,
    retry_on: tuple[type[BaseException], ...] = (OSError, TimeoutError),
) -> T:
    """Executa ``fn`` com retries exponenciais para falhas transitórias."""
    if attempts < 1:
        msg = "attempts deve ser >= 1"
        raise ValueError(msg)
    if not backoff_seconds:
        backoff_seconds = (0.0,)

    last_exc: BaseException | None = None
    for i in range(attempts):
        try:
            return fn()
        except retry_on as exc:
            last_exc = exc
            if i >= attempts - 1:
                break
            delay = backoff_seconds[i] if i < len(backoff_seconds) else backoff_seconds[-1]
            if delay > 0:
                factor = 1.0 + random.uniform(-jitter_ratio, jitter_ratio)
                time.sleep(max(0.0, delay * factor))
    assert last_exc is not None
    raise last_exc


@contextmanager
def file_lock(
    lock_path: Path,
    *,
    timeout_seconds: float = 20.0,
    poll_seconds: float = 0.1,
    stale_after_seconds: float = 900.0,
) -> Iterator[None]:
    """Lock simples por ficheiro com timeout e recuperação de lock stale."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"{os.getpid()}".encode())
            finally:
                os.close(fd)
            break
        except FileExistsError:
            if stale_after_seconds > 0 and lock_path.exists():
                age = time.time() - lock_path.stat().st_mtime
                if age > stale_after_seconds:
                    with suppress(OSError):
                        lock_path.unlink()
                    continue
            if time.monotonic() - start >= timeout_seconds:
                msg = f"Timeout ao aguardar lock: {lock_path}"
                raise TimeoutError(msg) from None
            time.sleep(poll_seconds)

    try:
        yield
    finally:
        with suppress(FileNotFoundError):
            lock_path.unlink()
