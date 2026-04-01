#!/usr/bin/env bash
# Run a simulation inside a Docker container.
#
# Usage:
#   ./scripts/simulate.sh                        # default scenario -> output/results.json
#   ./scripts/simulate.sh my_config.json          # custom config   -> output/results.json
#   ./scripts/simulate.sh my_config.json -o out.json  # custom output path (inside container)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

mkdir -p output

CONFIG_FILE="${1:-}"
shift 2>/dev/null || true

if [ -n "$CONFIG_FILE" ]; then
    # Mount the config file into the container
    CONFIG_BASENAME="$(basename "$CONFIG_FILE")"
    CONFIG_ABSPATH="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$CONFIG_BASENAME"

    docker compose run --rm \
        -v "$CONFIG_ABSPATH:/app/input/$CONFIG_BASENAME:ro" \
        simulator \
        simulate "/app/input/$CONFIG_BASENAME" -o /app/output/results.json "$@"
else
    docker compose run --rm simulator \
        simulate -o /app/output/results.json "$@"
fi

echo "Results written to output/results.json"
