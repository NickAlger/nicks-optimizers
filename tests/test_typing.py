"""Static typing catches mixed-up vectors and covectors: the snippets in
tests/typing_errors/ make deliberate mistakes, each marked with the error
code mypy --strict must report on its line, and mypy must report exactly
those (generic calls with swapped arguments fail type inference, code
"misc"). If the generic core lost its types (e.g. Any in lbfgs_memory), the
mistakes would pass and this test would fail. Skipped without mypy."""

import pathlib
import re

import pytest

api = pytest.importorskip("mypy.api")

ROOT = pathlib.Path(__file__).resolve().parent.parent
SNIPPETS = sorted((ROOT / "tests" / "typing_errors").glob("*.py"))
MARK = re.compile(r"\s#\s*E:\s*([a-z-]+)\s*$")
REPORTED = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+): error: .*\[(?P<code>[a-z-]+)\]$")


def expected() -> set[tuple[str, int, str]]:
    marks = set()
    for path in SNIPPETS:
        for number, text in enumerate(path.read_text().splitlines(), start=1):
            found = MARK.search(text)
            if found:
                marks.add((path.name, number, found.group(1)))
    return marks


def test_mixups_are_rejected_and_correct_lines_pass() -> None:
    assert SNIPPETS
    out, err, _ = api.run(["--config-file", str(ROOT / "pyproject.toml"), "--strict",
                           "--show-error-codes", "--no-error-summary", "--hide-error-context",
                           *map(str, SNIPPETS)])
    reported = set()
    for line in out.splitlines():
        match = REPORTED.match(line)
        assert match, f"unexpected mypy output: {line!r} ({err})"
        reported.add((pathlib.Path(match["file"]).name, int(match["line"]), match["code"]))
    assert reported == expected()
