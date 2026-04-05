"""Sensitivity analysis for ideological dynamics simulations.

Two complementary approaches:

1. **One-At-a-Time (OAT)**: vary each parameter individually across its
   range while holding all others at baseline. Reveals per-parameter
   response curves.

2. **Latin Hypercube Sampling (LHS)**: sample N points from the full
   parameter space using quasi-random LHS. Captures interaction effects
   and provides global summary statistics.

Both methods collect final ideology shares and health-check metrics for
each simulation run, packaging results into typed dataclasses that
serialize cleanly to JSON.

This module is pure computation -- no I/O, no CLI, no side effects.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.stats.qmc import LatinHypercube

from ideasim.engine import simulate
from ideasim.evaluation.fitting import (
    FittableParameter,
    _extract_parameters,
    _inject_parameters,
    build_parameter_spec,
)
from ideasim.health import HealthReport, check_health
from ideasim.types import SimulationConfig, SimulationResult

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    """Condensed output of a single simulation run.

    Attributes:
        parameter_values: Mapping of parameter name to the value used.
        final_shares: Mapping of ideology name to final share.
        health: HealthReport for the run.
        final_entropy: Shannon entropy of the final share distribution.
        n_events: Total number of stochastic events that occurred.
    """

    parameter_values: dict[str, float]
    final_shares: dict[str, float]
    health: HealthReport
    final_entropy: float
    n_events: int

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter_values": self.parameter_values,
            "final_shares": {k: round(v, 8) for k, v in self.final_shares.items()},
            "health": self.health.to_dict(),
            "final_entropy": round(self.final_entropy, 8),
            "n_events": self.n_events,
        }


@dataclass
class OATParameterResult:
    """OAT sweep result for a single parameter.

    Attributes:
        parameter_name: The parameter that was varied.
        values: The parameter values tested.
        runs: One RunSummary per value.
    """

    parameter_name: str
    values: list[float]
    runs: list[RunSummary]

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter_name": self.parameter_name,
            "values": [round(v, 8) for v in self.values],
            "runs": [r.to_dict() for r in self.runs],
        }


@dataclass
class LHSSummaryStats:
    """Aggregate statistics across all LHS samples for one output metric.

    Attributes:
        metric_name: e.g. "liberalism_final_share" or "final_entropy".
        mean: Mean across all samples.
        std: Standard deviation.
        min: Minimum.
        max: Maximum.
    """

    metric_name: str
    mean: float
    std: float
    min: float
    max: float

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "mean": round(self.mean, 8),
            "std": round(self.std, 8),
            "min": round(self.min, 8),
            "max": round(self.max, 8),
        }


@dataclass
class SensitivityResult:
    """Complete sensitivity analysis result.

    Attributes:
        oat: OAT results, one per parameter. Empty if OAT was not run.
        lhs_runs: Individual LHS run summaries. Empty if LHS was not run.
        lhs_stats: Summary statistics across LHS runs. Empty if LHS was not run.
        n_parameters: Number of parameters analyzed.
        metadata: Additional info (e.g. n_oat_steps, n_lhs_samples).
    """

    oat: list[OATParameterResult] = field(default_factory=list)
    lhs_runs: list[RunSummary] = field(default_factory=list)
    lhs_stats: list[LHSSummaryStats] = field(default_factory=list)
    n_parameters: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "oat": [o.to_dict() for o in self.oat],
            "lhs_runs": [r.to_dict() for r in self.lhs_runs],
            "lhs_stats": [s.to_dict() for s in self.lhs_stats],
            "n_parameters": self.n_parameters,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _summarize_run(
    result: SimulationResult,
    param_values: dict[str, float],
) -> RunSummary:
    """Distill a full SimulationResult into a lightweight RunSummary."""
    health = check_health(result)
    last = result.history[-1]

    # Final shares by ideology name
    final_shares: dict[str, float] = {}
    for name in result.ideology_names:
        idx = result.ideology_names.index(name)
        if idx < len(last.shares):
            final_shares[name] = float(last.shares[idx])

    # Shannon entropy
    x = last.shares
    x_pos = x[x > 1e-15]
    entropy = -float(np.sum(x_pos * np.log(x_pos)))

    return RunSummary(
        parameter_values=param_values,
        final_shares=final_shares,
        health=health,
        final_entropy=entropy,
        n_events=len(result.events),
    )


def _make_deterministic(config: SimulationConfig) -> SimulationConfig:
    """Return a deep copy with stochastic events disabled for reproducibility."""
    cfg = copy.deepcopy(config)
    cfg.stochastic.charisma_rate = 0.0
    cfg.stochastic.hybridization_rates[:] = 0.0
    for ideo in cfg.ideologies:
        ideo.mutation_rate = 0.0
    return cfg


def _run_with_values(
    base_config: SimulationConfig,
    params_spec: list[FittableParameter],
    values: NDArray[np.float64],
) -> RunSummary:
    """Inject parameter values, simulate, and return a summary."""
    cfg = _inject_parameters(base_config, params_spec, values)
    result = simulate(cfg)
    param_dict = {p.name: float(v) for p, v in zip(params_spec, values, strict=True)}
    return _summarize_run(result, param_dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_oat(
    config: SimulationConfig,
    n_steps: int = 10,
    deterministic: bool = True,
) -> list[OATParameterResult]:
    """One-At-a-Time sensitivity analysis.

    For each fittable parameter, holds all others at baseline and varies
    that parameter across ``n_steps`` evenly spaced values between its
    lower and upper bounds.

    Args:
        config: Baseline SimulationConfig.
        n_steps: Number of values to test per parameter (including endpoints).
        deterministic: If True, disable stochastic events for reproducibility.

    Returns:
        List of OATParameterResult, one per parameter.
    """
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")

    base = _make_deterministic(config) if deterministic else copy.deepcopy(config)
    params_spec = build_parameter_spec(base)
    baseline_values = _extract_parameters(base, params_spec)
    baseline_values = np.clip(
        baseline_values,
        np.array([p.low for p in params_spec]),
        np.array([p.high for p in params_spec]),
    )

    results: list[OATParameterResult] = []

    for idx, param in enumerate(params_spec):
        sweep_values = np.linspace(param.low, param.high, n_steps)
        runs: list[RunSummary] = []

        for val in sweep_values:
            trial_values = baseline_values.copy()
            trial_values[idx] = val
            summary = _run_with_values(base, params_spec, trial_values)
            runs.append(summary)

        results.append(
            OATParameterResult(
                parameter_name=param.name,
                values=sweep_values.tolist(),
                runs=runs,
            )
        )

    return results


def run_lhs(
    config: SimulationConfig,
    n_samples: int = 50,
    seed: int | None = 42,
    deterministic: bool = True,
) -> tuple[list[RunSummary], list[LHSSummaryStats]]:
    """Latin Hypercube Sampling sensitivity analysis.

    Draws ``n_samples`` parameter combinations that cover the full
    parameter space quasi-randomly, runs the simulation for each, and
    computes summary statistics across all runs.

    Args:
        config: Baseline SimulationConfig (provides structure and bounds).
        n_samples: Number of LHS samples to draw.
        seed: Random seed for LHS sampling reproducibility.
        deterministic: If True, disable stochastic events for reproducibility.

    Returns:
        Tuple of (list of RunSummary, list of LHSSummaryStats).
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    base = _make_deterministic(config) if deterministic else copy.deepcopy(config)
    params_spec = build_parameter_spec(base)
    n_params = len(params_spec)

    lows = np.array([p.low for p in params_spec], dtype=np.float64)
    highs = np.array([p.high for p in params_spec], dtype=np.float64)

    # Generate LHS samples in [0, 1]^d and scale to parameter bounds
    sampler = LatinHypercube(d=n_params, seed=seed)
    unit_samples = sampler.random(n=n_samples)  # shape (n_samples, n_params)
    scaled_samples = lows + unit_samples * (highs - lows)

    runs: list[RunSummary] = []
    for i in range(n_samples):
        summary = _run_with_values(base, params_spec, scaled_samples[i])
        runs.append(summary)

    # Compute summary statistics
    stats = _compute_lhs_stats(runs)

    return runs, stats


def _compute_lhs_stats(runs: list[RunSummary]) -> list[LHSSummaryStats]:
    """Compute aggregate statistics across LHS runs."""
    if not runs:
        return []

    # Collect all ideology names from first run
    ideology_names = list(runs[0].final_shares.keys())

    stats: list[LHSSummaryStats] = []

    # Per-ideology final share stats
    for name in ideology_names:
        values = np.array([r.final_shares.get(name, 0.0) for r in runs])
        stats.append(
            LHSSummaryStats(
                metric_name=f"{name}_final_share",
                mean=float(np.mean(values)),
                std=float(np.std(values)),
                min=float(np.min(values)),
                max=float(np.max(values)),
            )
        )

    # Entropy stats
    entropies = np.array([r.final_entropy for r in runs])
    stats.append(
        LHSSummaryStats(
            metric_name="final_entropy",
            mean=float(np.mean(entropies)),
            std=float(np.std(entropies)),
            min=float(np.min(entropies)),
            max=float(np.max(entropies)),
        )
    )

    # Health pass rate
    health_passes = np.array([1.0 if r.health.all_passed else 0.0 for r in runs])
    stats.append(
        LHSSummaryStats(
            metric_name="health_pass_rate",
            mean=float(np.mean(health_passes)),
            std=float(np.std(health_passes)),
            min=float(np.min(health_passes)),
            max=float(np.max(health_passes)),
        )
    )

    # Event count stats
    event_counts = np.array([float(r.n_events) for r in runs])
    stats.append(
        LHSSummaryStats(
            metric_name="n_events",
            mean=float(np.mean(event_counts)),
            std=float(np.std(event_counts)),
            min=float(np.min(event_counts)),
            max=float(np.max(event_counts)),
        )
    )

    return stats


def run_sensitivity(
    config: SimulationConfig,
    oat_steps: int = 10,
    lhs_samples: int = 50,
    seed: int | None = 42,
    deterministic: bool = True,
    run_oat_analysis: bool = True,
    run_lhs_analysis: bool = True,
) -> SensitivityResult:
    """Run full sensitivity analysis combining OAT and LHS.

    This is the main entry point. Runs both analyses (or either one
    independently) and packages everything into a single result object.

    Args:
        config: Baseline SimulationConfig.
        oat_steps: Number of values per parameter in OAT sweep.
        lhs_samples: Number of LHS samples.
        seed: Random seed for LHS.
        deterministic: Disable stochastic events for reproducibility.
        run_oat_analysis: Whether to run OAT analysis.
        run_lhs_analysis: Whether to run LHS analysis.

    Returns:
        SensitivityResult with all collected data.
    """
    params_spec = build_parameter_spec(config)
    n_params = len(params_spec)

    result = SensitivityResult(
        n_parameters=n_params,
        metadata={
            "parameter_names": [p.name for p in params_spec],
            "oat_steps": oat_steps if run_oat_analysis else 0,
            "lhs_samples": lhs_samples if run_lhs_analysis else 0,
            "deterministic": deterministic,
            "seed": seed,
        },
    )

    if run_oat_analysis:
        result.oat = run_oat(
            config,
            n_steps=oat_steps,
            deterministic=deterministic,
        )

    if run_lhs_analysis:
        lhs_runs, lhs_stats = run_lhs(
            config,
            n_samples=lhs_samples,
            seed=seed,
            deterministic=deterministic,
        )
        result.lhs_runs = lhs_runs
        result.lhs_stats = lhs_stats

    return result
