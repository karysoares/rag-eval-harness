"""Cobertura de resiliência: retry_call e file_lock."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from llm_evaluation.resilience import file_lock, retry_call


class TestRetryCall:
    def test_success_first_attempt(self) -> None:
        calls: list[int] = []

        def fn() -> str:
            calls.append(1)
            return "ok"

        assert retry_call(fn) == "ok"
        assert len(calls) == 1

    def test_retries_then_succeeds(self) -> None:
        attempts: list[int] = []

        def fn() -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise OSError("transient")
            return "ok"

        out = retry_call(fn, attempts=3, backoff_seconds=(0.0, 0.0))
        assert out == "ok"
        assert len(attempts) == 3

    def test_exhausts_attempts_and_raises_last(self) -> None:
        def fn() -> None:
            raise TimeoutError("always")

        with pytest.raises(TimeoutError, match="always"):
            retry_call(fn, attempts=2, backoff_seconds=(0.0,))

    def test_non_retryable_exception_propagates_immediately(self) -> None:
        attempts: list[int] = []

        def fn() -> None:
            attempts.append(1)
            raise ValueError("logic bug")

        with pytest.raises(ValueError, match="logic bug"):
            retry_call(fn, attempts=5, backoff_seconds=(0.0,))
        assert len(attempts) == 1

    def test_invalid_attempts_rejected(self) -> None:
        with pytest.raises(ValueError, match="attempts"):
            retry_call(lambda: 1, attempts=0)

    def test_empty_backoff_tuple_defaults_to_zero_delay(self) -> None:
        attempts: list[int] = []

        def fn() -> str:
            attempts.append(1)
            if len(attempts) < 2:
                raise OSError("x")
            return "ok"

        assert retry_call(fn, attempts=2, backoff_seconds=()) == "ok"

    def test_backoff_shorter_than_attempts_reuses_last_delay(self) -> None:
        attempts: list[int] = []

        def fn() -> str:
            attempts.append(1)
            if len(attempts) < 4:
                raise OSError("x")
            return "ok"

        start = time.monotonic()
        out = retry_call(fn, attempts=4, backoff_seconds=(0.01,), jitter_ratio=0.0)
        elapsed = time.monotonic() - start
        assert out == "ok"
        assert elapsed >= 0.02  # 3 sleeps de ~0.01s cada


class TestFileLock:
    def test_acquire_and_release(self, tmp_path: Path) -> None:
        lock = tmp_path / "x.lock"
        with file_lock(lock):
            assert lock.exists()
        assert not lock.exists()

    def test_lock_file_records_pid(self, tmp_path: Path) -> None:
        lock = tmp_path / "x.lock"
        with file_lock(lock):
            content = lock.read_text()
            assert content.isdigit()

    def test_contention_times_out(self, tmp_path: Path) -> None:
        lock = tmp_path / "x.lock"
        with file_lock(lock), pytest.raises(TimeoutError, match="lock"):
            file_lock(
                lock,
                timeout_seconds=0.3,
                poll_seconds=0.05,
                stale_after_seconds=0,
            ).__enter__()

    def test_stale_lock_is_recovered(self, tmp_path: Path) -> None:
        lock = tmp_path / "x.lock"
        lock.write_text("999999")
        old = time.time() - 3600
        import os

        os.utime(lock, (old, old))
        with file_lock(lock, timeout_seconds=2.0, stale_after_seconds=60.0):
            assert lock.exists()
        assert not lock.exists()

    def test_sequential_holders(self, tmp_path: Path) -> None:
        lock = tmp_path / "x.lock"
        order: list[str] = []
        # Uma excepção numa thread não chega ao teste: sem a recolher, um timeout
        # do lock sob carga aparecia como IndexError na asserção final, a apontar
        # para o sítio errado. O timeout é generoso porque a máquina pode estar a
        # correr outra coisa — o que se mede é exclusão mútua, não latência.
        erros: list[BaseException] = []

        def worker(tag: str) -> None:
            try:
                with file_lock(lock, timeout_seconds=30.0, poll_seconds=0.01):
                    order.append(f"{tag}-in")
                    time.sleep(0.05)
                    order.append(f"{tag}-out")
            except BaseException as exc:  # noqa: BLE001 — recolher para reportar
                erros.append(exc)

        t1 = threading.Thread(target=worker, args=("a",))
        t2 = threading.Thread(target=worker, args=("b",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not erros, f"worker falhou: {erros!r}"
        assert len(order) == 4, f"esperadas 4 transições, obtidas {order!r}"
        # Secções críticas nunca se sobrepõem: cada "-in" é seguido do próprio "-out".
        assert order[0].split("-")[0] == order[1].split("-")[0]
        assert order[2].split("-")[0] == order[3].split("-")[0]
