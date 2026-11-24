"""The documentation's examples run as written.

Every ``.. code-block:: python`` block in ``docs/*.rst`` and every python fence
in ``README.md`` is executed in a fresh temporary directory. Every command line
shown that starts with ``ideasim`` is parsed with the CLI's own parser (not
run), so a renamed subcommand or option fails here.

A python example that needs the network or minutes of CPU is marked in the
source, on the line before it, and then runs only with ``--run-slow``::

    .. docs-test: slow                 (reStructuredText)
    <!-- docs-test: slow -->           (Markdown)
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

import pytest

from ideasim.cli import build_parser

REPO = Path(__file__).resolve().parents[1]
SLOW_MARK_RST = ".. docs-test: slow"
SLOW_MARK_MD = "<!-- docs-test: slow -->"


@dataclass(frozen=True)
class Snippet:
    """One example taken from a documentation file."""

    source: str  # "docs/quickstart.rst:26"
    kind: str  # "python" or "shell"
    code: str
    slow: bool


def _dedent_block(lines: list[str], start: int) -> tuple[list[str], int]:
    """Collect the indented body of an RST block starting at ``start``.

    Returns the dedented body lines and the index of the first line after it.
    """
    i = start
    while i < len(lines) and not lines[i].strip():
        i += 1
    body: list[str] = []
    indent: int | None = None
    while i < len(lines):
        line = lines[i]
        if line.strip():
            current = len(line) - len(line.lstrip())
            if indent is None:
                indent = current
            if current < indent or current == 0:
                break
            body.append(line[indent:])
        else:
            body.append("")
        i += 1
    while body and not body[-1]:
        body.pop()
    return body, i


def _previous_nonblank(lines: list[str], i: int) -> str:
    j = i - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    return lines[j].strip() if j >= 0 else ""


def extract_rst(text: str, name: str) -> list[Snippet]:
    """Python code blocks and shell literal blocks of an RST document."""
    lines = text.splitlines()
    snippets: list[Snippet] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        directive = re.match(r"\.\. code-block::\s*(\S+)", stripped)
        if directive:
            language = directive.group(1)
            body, end = _dedent_block(lines, i + 1)
            # options such as :linenos: sit directly under the directive
            while body and body[0].startswith(":"):
                body.pop(0)
            kind = "python" if language == "python" else "shell"
            if language in ("python", "bash", "console", "shell", "sh"):
                snippets.append(
                    Snippet(
                        f"{name}:{i + 1}",
                        kind,
                        "\n".join(body) + "\n",
                        _previous_nonblank(lines, i) == SLOW_MARK_RST,
                    )
                )
            i = end
            continue
        if stripped.endswith("::") and not stripped.startswith(".."):
            body, end = _dedent_block(lines, i + 1)
            first = i + 1
            while first < len(lines) and not lines[first].strip():
                first += 1
            snippets.append(
                Snippet(f"{name}:{first + 1}", "shell", "\n".join(body), False)
            )
            i = end
            continue
        i += 1
    return snippets


def extract_markdown(text: str, name: str) -> list[Snippet]:
    """Python and shell fences of a Markdown document."""
    lines = text.splitlines()
    snippets: list[Snippet] = []
    i = 0
    while i < len(lines):
        fence = re.match(r"```(\w*)\s*$", lines[i].strip())
        if not fence:
            i += 1
            continue
        language = fence.group(1)
        j = i + 1
        while j < len(lines) and lines[j].strip() != "```":
            j += 1
        body = "\n".join(lines[i + 1 : j]) + "\n"
        if language == "python":
            slow = _previous_nonblank(lines, i) == SLOW_MARK_MD
            snippets.append(Snippet(f"{name}:{i + 1}", "python", body, slow))
        elif language in ("bash", "console", "shell", "sh"):
            snippets.append(Snippet(f"{name}:{i + 1}", "shell", body, False))
        i = j + 1
    return snippets


def collect() -> list[Snippet]:
    snippets: list[Snippet] = []
    for path in sorted((REPO / "docs").glob("*.rst")):
        name = path.relative_to(REPO).as_posix()
        snippets += extract_rst(path.read_text(encoding="utf-8"), name)
    readme = REPO / "README.md"
    snippets += extract_markdown(readme.read_text(encoding="utf-8"), "README.md")
    return snippets


def cli_lines(snippet: Snippet) -> list[list[str]]:
    """The ``ideasim ...`` command lines of a shell snippet, as argv lists."""
    commands = []
    for raw in snippet.code.splitlines():
        line = raw.strip().removeprefix("$ ")
        if line.startswith("ideasim "):
            commands.append(shlex.split(line, comments=True)[1:])
    return commands


SNIPPETS = collect()
PYTHON = [s for s in SNIPPETS if s.kind == "python"]
COMMANDS = [(s.source, argv) for s in SNIPPETS for argv in cli_lines(s)]


def test_examples_were_found() -> None:
    """An extractor that silently finds nothing would make every test pass."""
    assert len(PYTHON) >= 3, [s.source for s in PYTHON]
    assert len(COMMANDS) >= 5, COMMANDS


@pytest.mark.parametrize(
    "snippet",
    [
        pytest.param(s, id=s.source, marks=[pytest.mark.slow] if s.slow else [])
        for s in PYTHON
    ],
)
def test_python_example_runs(
    snippet: Snippet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    code = compile(snippet.code, snippet.source, "exec")
    exec(code, {"__name__": "__docs_example__"})


@pytest.mark.parametrize(
    ("source", "argv"),
    [pytest.param(src, argv, id=f"{src} {' '.join(argv)}") for src, argv in COMMANDS],
)
def test_cli_example_parses(source: str, argv: list[str]) -> None:
    try:
        build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse reports a bad command this way
        pytest.fail(f"{source}: 'ideasim {' '.join(argv)}' does not parse ({exc})")


class TestExtractors:
    """The extractors themselves, on documents whose content is known."""

    def test_rst_python_block_and_slow_mark(self) -> None:
        text = (
            "Title\n=====\n\n.. docs-test: slow\n\n.. code-block:: python\n\n"
            "    x = 1\n    if x:\n        y = 2\n\nAfter.\n"
        )
        (snippet,) = extract_rst(text, "t.rst")
        assert snippet.kind == "python"
        assert snippet.code == "x = 1\nif x:\n    y = 2\n"
        assert snippet.slow
        assert snippet.source == "t.rst:6"

    def test_rst_literal_block_is_shell(self) -> None:
        text = "Using the CLI::\n\n    ideasim simulate -o r.json\n\nNext.\n"
        (snippet,) = extract_rst(text, "t.rst")
        assert cli_lines(snippet) == [["simulate", "-o", "r.json"]]

    def test_markdown_fences(self) -> None:
        text = (
            "<!-- docs-test: slow -->\n```python\nprint(1)\n```\n\n"
            "```bash\n# comment\nideasim validate cfg.json  # trailing\n```\n"
            "```\nideasim not-a-command\n```\n"
        )
        python, shell = extract_markdown(text, "R.md")
        assert python.slow and python.code == "print(1)\n"
        assert cli_lines(shell) == [["validate", "cfg.json"]]

    def test_bad_command_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["simulate", "--no-such-option"])
