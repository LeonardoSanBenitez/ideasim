#!/usr/bin/env bash
# Run the test suite inside a Docker container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

docker compose run --rm --no-deps --entrypoint python simulator \
    -m pytest tests/ -v "$@"
