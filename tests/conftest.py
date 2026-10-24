"""On GitHub Actions, report test failures as annotations, so they appear on
the run's page (and through the API) without opening the log."""

import os

import pytest


def _escape_message(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    return _escape_message(text).replace(":", "%3A").replace(",", "%2C")


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    for report in terminalreporter.stats.get("failed", []):
        tail = "\n".join(str(report.longrepr).splitlines()[-40:])
        terminalreporter.write_line(
            f"::error title={_escape_property('failed: ' + report.nodeid)}::{_escape_message(tail)}")
