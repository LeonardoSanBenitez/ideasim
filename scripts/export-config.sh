#!/usr/bin/env bash
# Export the default simulation config to a JSON file.
#
# Usage:
#   ./scripts/export-config.sh                    # -> output/default_config.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

mkdir -p output

docker compose run --rm simulator \
    default-config -o /app/output/default_config.json

echo "Default config written to output/default_config.json"
