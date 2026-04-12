# ideasim

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python versions](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://github.com/LeonardoSanBenitez/ideasim)
[![Tests](https://github.com/LeonardoSanBenitez/ideasim/actions/workflows/test.yml/badge.svg)](https://github.com/LeonardoSanBenitez/ideasim/actions/workflows/test.yml)

A simulator for ideological competition dynamics over historical timescales.
Models how ideological currents compete, spread, go dormant, and re-emerge
using replicator-mutator dynamics with dormancy, hybridization, and endogenous
environment modification. The full system is a Piecewise Deterministic Markov
Process (PDMP).

**Disclaimer: this projects makes heavy use of AI assistants, from ideation to implementation.**

## Quick Start

```bash
pip install ideasim
```

Run the default scenario (Europe 1800-2025):

```bash
ideasim simulate -o results.json
```

Export and edit the default configuration:

```bash
ideasim default-config -o my_config.json
# edit my_config.json ...
ideasim simulate my_config.json -o results.json
```

## What It Models

`ideasim` simulates how ideological currents (liberalism, Marxism, traditionalism,
nationalism, and their hybrids) evolve over time. The model captures:

- **Manifest adoption** (x): observable adherence to ideologies on the simplex.
- **Latent sympathy** (y): hidden reservoirs of support that enable dormancy
  and re-emergence (the "martyrdom effect").
- **Environment** (e): exogenous conditions (GDP growth, inequality, media
  freedom, social media penetration, conflict) that shape ideological fitness.
- **Stochastic events**: hybridization (new ideologies from blending two
  parents), charismatic bursts, and parameter mutation.

For the full mathematical framework, see the `paper/` directory or
[the documentation](https://leonardosanbenitez.github.io/ideasim/).

## Installation

### From PyPI

```bash
pip install ideasim
```

### From source (development)

```bash
git clone https://github.com/LeonardoSanBenitez/ideasim.git
cd ideasim
pip install -e ".[dev]"
```

### Docker

```bash
docker compose build
docker compose run --rm simulator
```

## Usage

### Python API

```python
from ideasim.defaults import load_default_config
from ideasim.engine import simulate

config = load_default_config()
result = simulate(config, seed=42)

print(f"Final shares: {result.final_shares}")
print(f"Events triggered: {len(result.events)}")
```

### CLI

```bash
# Run with default parameters
ideasim simulate -o results.json

# Custom config
ideasim simulate my_config.json -o results.json

# Export default config for editing
ideasim default-config -o default_config.json
```

### Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
ideasim/
    ideasim/             # Python package
        types.py         # Core data types (Ideology, SimulationConfig, etc.)
        engine.py        # PDMP simulation engine (RK4 + stochastic events)
        defaults.py      # Loads default scenario from JSON data file
        io.py            # JSON serialization/deserialization
        cli.py           # Command-line interface
        sensitivity.py   # Sensitivity analysis (OAT + LHS)
        validation.py    # Input validation
        health.py        # Health checks
        evaluation/      # Validation against observed data (ParlGov)
        data/
            default_scenario.json
    tests/               # Test suite (104 tests)
    paper/               # LaTeX paper
    scripts/             # Shell scripts for Docker workflows
```

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines on setting up the development environment, running tests, and
submitting pull requests.

## Citation

If you use `ideasim` in your research, please cite:

```bibtex
@software{ideasim,
  title = {ideasim: Ideological Dynamics Simulator},
  author = {San Benitez, Leonardo},
  year = {2026},
  url = {https://github.com/LeonardoSanBenitez/ideasim},
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
