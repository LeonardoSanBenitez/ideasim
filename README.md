# ideasim -- Ideological Dynamics Simulator

A simulation of ideological competition over historical timescales, based on
replicator-mutator dynamics with dormancy, hybridization, and endogenous
environment modification. The full system is a Piecewise Deterministic Markov
Process (PDMP).

## Overview

`ideasim` models how ideological currents (liberalism, Marxism, traditionalism,
nationalism, and their hybrids) compete, spread, go dormant, and re-emerge over
time. The model captures:

- **Manifest adoption** (x): observable adherence to ideologies (on the simplex).
- **Latent sympathy** (y): hidden reservoirs of support that enable dormancy
  and re-emergence.
- **Environment** (e): exogenous conditions (GDP growth, inequality, media
  freedom, social media penetration, conflict) that shape ideological fitness.
- **Stochastic events**: hybridization (new ideologies from blending two
  parents), charismatic bursts, and parameter mutation.

## Mathematical Framework

The continuous dynamics follow a replicator-mutator equation with
latent-driven immigration:

    dx_i/dt = x_i * (f_i - f_bar) + rho * (y_i - x_i * Y)

where f_i is the fitness of ideology i (environmental response + competitive
interaction + institutional capture + latent reservoir), and rho controls
latent-to-manifest coupling.

Latent sympathy evolves via:

    dy_i/dt = alpha_i * x_i - beta_i * y_i + sigma_i * (1 - x_i) + delta_i * S_i

where S_i is the suppression pressure on ideology i (martyrdom effect).

Stochastic events overlay the continuous dynamics as a Poisson process.

See the `paper/` directory for the full formal treatment.

## Installation

Requires Python >= 3.11 and numpy >= 1.26.

    cd ideasim
    pip install -e ".[dev]"

## Usage

### Run the default scenario (Europe 1800-2025)

    ideasim simulate -o results.json

### Run with a custom config

    ideasim simulate my_config.json -o results.json

### Export the default config (for editing)

    ideasim default-config -o default_config.json

### Run tests

    python3 -m pytest tests/ -v

### Docker

    docker compose build
    docker compose run --rm simulator

Results are written to `output/results.json`.

## Project Structure

    ideasim/
        ideasim/           # Python package
            __init__.py
            types.py       # Core data types (Ideology, SimulationConfig, etc.)
            engine.py      # PDMP simulation engine (RK4 + stochastic events)
            defaults.py    # Default European 1800-2025 scenario parameters
            io.py          # JSON serialization/deserialization
            cli.py         # Command-line interface
        tests/
            test_engine.py # Test suite (simplex invariance, dormancy, reproducibility)
        paper/             # LaTeX paper (introduction, related work, references)
        scripts/           # Shell scripts for Docker workflows
        Dockerfile
        docker-compose.yml
        pyproject.toml

## Status

The simulation runs end-to-end and produces deterministic results. The default
scenario parameters need tuning -- traditionalism currently dominates at 84%,
which is not historically realistic. Next steps:

1. Parameter tuning for historical plausibility
2. Reporting layer (visualization, narrative summaries)
3. Validation against historical data (European populism 2008-2025)
4. Paper completion

## Team

- **stephan** -- architecture, types, integration, CLI, JSON I/O
- **lucas** -- simulation engine (PDMP integrator, core loop)
- **cidral** -- mathematical formalization and validation
- **mark** -- data calibration, reporting, historical validation
- **jose** -- conceptual framework (ideology mechanisms)
