"""Default scenario: 4 ideologies in Europe, 1800-2025.

Parameters from Cidral's formalization (Section 7.3-7.6),
validated against Jose's mechanisms and Mark's historical cases.

The default scenario is stored as a JSON file in data/default_scenario.json.
This module loads that file and exposes the same API as before.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from ideasim.io import load_config
from ideasim.types import Ideology, SimulationConfig


def _default_scenario_path() -> Path:
    """Return the path to the bundled default scenario JSON."""
    return Path(resources.files("ideasim") / "data" / "default_scenario.json")  # type: ignore[arg-type]


def make_default_ideologies() -> list[Ideology]:
    """Create the default 4-ideology system."""
    config = load_config(_default_scenario_path())
    return config.ideologies


def make_default_config() -> SimulationConfig:
    """Create the default European 1800-2025 simulation config."""
    return load_config(_default_scenario_path())
