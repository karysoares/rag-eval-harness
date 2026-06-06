"""Fixtures e hooks partilhados pela suite de testes."""

from __future__ import annotations

import os

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Testes ``@pytest.mark.integration`` só correm com ``RUN_INTEGRATION=1``."""
    if item.get_closest_marker("integration") is None:
        return
    if os.environ.get("RUN_INTEGRATION", "0") != "1":
        pytest.skip("Defina RUN_INTEGRATION=1 para executar testes de integração")
