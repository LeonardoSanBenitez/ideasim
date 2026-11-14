# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `fit_parameters(..., free=[...])` fits only the named parameters and keeps
  the rest at their configured values; `select_parameters` validates the names.

### Fixed

- With `align_t_start=True`, the fitted `best_config` keeps the aligned start
  year its initial shares and latent values were fitted at. This changes
  reported training metrics and saved fits made with that option.

## [0.1.0] - 2026-04-03

### Added

- PDMP simulation engine with RK4 integrator and stochastic event overlay.
- Replicator-mutator dynamics with latent sympathy (dormancy/martyrdom).
- Endogenous environment modification.
- Hybridization as stochastic jump process.
- Default scenario: European ideological competition 1800-2025.
- CLI with `ideasim simulate` and `ideasim default-config` commands.
- JSON serialization/deserialization for configs and results.
- Sensitivity analysis module (OAT sweeps and Latin Hypercube Sampling).
- Input validation and health checks.
- Evaluation framework for validation against ParlGov observed data.
- Docker support.
- 104 tests covering engine, CLI, evaluation, health, validation, and sensitivity.
