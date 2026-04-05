# Contributing to ideasim

Thank you for your interest in contributing to ideasim.

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/LeonardoSanBenitez/ideasim.git
cd ideasim
```

2. Create a virtual environment and install in development mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

3. (Optional) Install pre-commit hooks:

```bash
pre-commit install
```

## Running Tests

```bash
python -m pytest tests/ -v
```

To run slow tests (e.g., ParlGov data fetching):

```bash
python -m pytest tests/ -v --run-slow
```

## Code Style

- All code must have type annotations and pass `mypy --strict`.
- Follow PEP 8 conventions. We use `ruff` for linting and formatting.
- Every new feature or bugfix must include tests.

Format and lint your code before committing:

```bash
ruff check . --fix
ruff format .
mypy ideasim/
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Write your changes with tests and type annotations.
3. Ensure all tests pass and linting is clean.
4. Write a clear PR description explaining what changed and why.
5. Submit the pull request against `main`.

## Reporting Issues

- Use the GitHub issue tracker.
- For bugs, include: Python version, OS, minimal reproduction steps,
  expected vs actual behavior.
- For feature requests, describe the use case and proposed API.
