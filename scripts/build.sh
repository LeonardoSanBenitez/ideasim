#!/usr/bin/env bash
# Build the Docker image for the Ideological Dynamics Simulator.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

docker compose build
echo "Build complete."
