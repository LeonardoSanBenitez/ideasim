"""Parameter fitting: optimize simulation config against observed data.

Uses scipy.optimize.differential_evolution for global optimization of
simulation parameters to minimize MSE between simulated and observed
ideology shares on the training set.

The fitting operates on a flattened parameter vector extracted from
SimulationConfig. Only a selected subset of parameters is optimized
(those most likely to need tuning); the rest remain fixed.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, differential_evolution, OptimizeResult

from ideasim.engine import simulate
from ideasim.evaluation.dataset import ObservedDataset
from ideasim.evaluation.metrics import compute_metrics, EvaluationMetrics
from ideasim.types import SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class FittableParameter:
    """Describes one tunable parameter with bounds.

    Attributes:
        name: Human-readable name (e.g., "liberalism.latent_conversion").
        extract: Callable that gets the value from a SimulationConfig.
        inject: Callable that sets the value on a SimulationConfig.
        low: Lower bound.
        high: Upper bound.
    """

    name: str
    low: float
    high: float
    # These will be set by _build_parameter_spec
    _ideology_idx: int | None = None
    _param_name: str | None = None


@dataclass
class FitResult:
    """Result of a parameter fitting run.

    Attributes:
        best_config: The SimulationConfig with optimized parameters.
        train_metrics: Metrics on the training set.
        parameter_values: Dict of parameter_name -> fitted value.
        optimization_result: Raw scipy OptimizeResult (for diagnostics).
        n_evaluations: Number of objective function evaluations.
        metadata: Additional info.
    """

    best_config: SimulationConfig
    train_metrics: EvaluationMetrics
    parameter_values: dict[str, float]
    n_evaluations: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict (excludes config object)."""
        return {
            "train_metrics": self.train_metrics.to_dict(),
            "parameter_values": {
                k: round(v, 8) for k, v in self.parameter_values.items()
            },
            "n_evaluations": self.n_evaluations,
            "metadata": self.metadata,
        }


# --- Parameter extraction / injection ---

# We tune per-ideology parameters and a few global ones.
# Each parameter is identified by (ideology_index, param_name) or
# (None, global_param_name).

_IDEOLOGY_PARAMS: list[tuple[str, float, float]] = [
    # (param_name, low_bound, high_bound)
    ("latent_conversion", 0.001, 0.5),
    ("latent_generation", 0.01, 1.0),
    ("latent_decay", 0.001, 0.3),
    ("martyrdom_coeff", 0.01, 2.0),
    ("institutional_amp", 0.01, 1.0),
    ("capture_threshold", 0.05, 0.6),
    ("governance_threshold", 0.1, 0.8),
]

_GLOBAL_PARAMS: list[tuple[str, float, float]] = [
    ("rho", 0.0001, 0.1),
]


def build_parameter_spec(
    config: SimulationConfig,
) -> list[FittableParameter]:
    """Build the list of fittable parameters for a given config.

    Returns one FittableParameter per tunable scalar in the config.
    """
    params: list[FittableParameter] = []

    for i, ideo in enumerate(config.ideologies):
        for param_name, low, high in _IDEOLOGY_PARAMS:
            params.append(
                FittableParameter(
                    name=f"{ideo.name}.{param_name}",
                    low=low,
                    high=high,
                    _ideology_idx=i,
                    _param_name=param_name,
                )
            )

    for param_name, low, high in _GLOBAL_PARAMS:
        params.append(
            FittableParameter(
                name=param_name,
                low=low,
                high=high,
                _ideology_idx=None,
                _param_name=param_name,
            )
        )

    return params


def _inject_parameters(
    config: SimulationConfig,
    params_spec: list[FittableParameter],
    values: NDArray[np.float64],
) -> SimulationConfig:
    """Create a new config with the given parameter values injected."""
    new_config = copy.deepcopy(config)

    for param, value in zip(params_spec, values):
        if param._ideology_idx is not None and param._param_name is not None:
            setattr(
                new_config.ideologies[param._ideology_idx],
                param._param_name,
                float(value),
            )
        elif param._param_name is not None:
            setattr(new_config, param._param_name, float(value))

    return new_config


def _extract_parameters(
    config: SimulationConfig,
    params_spec: list[FittableParameter],
) -> NDArray[np.float64]:
    """Extract current parameter values from a config."""
    values = np.zeros(len(params_spec), dtype=np.float64)

    for i, param in enumerate(params_spec):
        if param._ideology_idx is not None and param._param_name is not None:
            values[i] = getattr(
                config.ideologies[param._ideology_idx],
                param._param_name,
            )
        elif param._param_name is not None:
            values[i] = getattr(config, param._param_name)

    return values


def _simulate_and_extract(
    config: SimulationConfig,
    observed_years: NDArray[np.float64],
    sim_ideology_names: list[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run simulation and extract shares at observed years.

    Returns (sim_years, sim_shares) arrays aligned with sim_ideology_names.
    """
    result = simulate(config)
    sim_times = result.times()
    n_ideo = len(sim_ideology_names)
    sim_shares = np.zeros((len(sim_times), n_ideo), dtype=np.float64)

    for i, name in enumerate(sim_ideology_names):
        if name in result.ideology_names:
            sim_shares[:, i] = result.share_timeseries(name)

    return sim_times, sim_shares


def _objective(
    values: NDArray[np.float64],
    base_config: SimulationConfig,
    params_spec: list[FittableParameter],
    train_dataset: ObservedDataset,
    sim_ideology_names: list[str],
    observed_years: NDArray[np.float64],
    observed_shares: NDArray[np.float64],
) -> float:
    """Objective function: MSE between simulation and training data."""
    try:
        candidate_config = _inject_parameters(base_config, params_spec, values)
        # Ensure simulation covers the training period
        candidate_config.t_end = float(observed_years[-1]) + 1.0

        sim_times, sim_shares = _simulate_and_extract(
            candidate_config, observed_years, sim_ideology_names
        )

        metrics = compute_metrics(
            observed_years=observed_years,
            observed_shares=observed_shares,
            simulated_years=sim_times,
            simulated_shares=sim_shares,
            ideology_names=sim_ideology_names,
        )

        if np.isnan(metrics.mse):
            return 1e6

        return metrics.mse

    except Exception as exc:
        logger.debug("Simulation failed for candidate: %s", exc)
        return 1e6


def fit_parameters(
    base_config: SimulationConfig,
    train_dataset: ObservedDataset,
    max_iterations: int = 100,
    population_size: int = 15,
    seed: int | None = 42,
    tolerance: float = 1e-6,
    workers: int = 1,
) -> FitResult:
    """Fit simulation parameters to observed training data.

    Uses differential evolution (a global optimizer) to find parameters
    that minimize MSE between simulated and observed ideology shares
    on the training set.

    Args:
        base_config: Starting SimulationConfig (provides structure and
            initial parameter values).
        train_dataset: Observed data to fit against.
        max_iterations: Maximum optimizer iterations.
        population_size: DE population size multiplier.
        seed: Random seed for the optimizer.
        tolerance: Convergence tolerance.
        workers: Number of parallel workers (1 = serial).

    Returns:
        FitResult with optimized config and training metrics.
    """
    sim_ideology_names = base_config.ideology_names()
    observed_years, observed_shares = train_dataset.as_matrix(sim_ideology_names)

    if len(observed_years) == 0:
        raise ValueError("Training dataset has no observations.")

    params_spec = build_parameter_spec(base_config)
    bounds = Bounds(
        lb=np.array([p.low for p in params_spec], dtype=np.float64),
        ub=np.array([p.high for p in params_spec], dtype=np.float64),
    )

    # Use current config values as initial guess (inserted into population)
    x0 = _extract_parameters(base_config, params_spec)
    # Clip to bounds
    x0 = np.clip(x0, bounds.lb, bounds.ub)

    logger.info(
        "Fitting %d parameters against %d observations (%d years, %d ideologies)",
        len(params_spec),
        int(np.sum(~np.isnan(observed_shares))),
        len(observed_years),
        len(sim_ideology_names),
    )

    # Disable stochastic events during fitting for determinism
    fit_config = copy.deepcopy(base_config)
    fit_config.seed = seed if seed is not None else 42
    fit_config.stochastic.charisma_rate = 0.0
    fit_config.stochastic.hybridization_rates[:] = 0.0
    for ideo in fit_config.ideologies:
        ideo.mutation_rate = 0.0

    result: OptimizeResult = differential_evolution(
        func=_objective,
        bounds=bounds,
        args=(
            fit_config,
            params_spec,
            train_dataset,
            sim_ideology_names,
            observed_years,
            observed_shares,
        ),
        maxiter=max_iterations,
        popsize=population_size,
        seed=seed,
        tol=tolerance,
        x0=x0,
        workers=workers,
        disp=False,
    )

    # Build the best config
    best_config = _inject_parameters(base_config, params_spec, result.x)
    param_values = {
        p.name: float(v) for p, v in zip(params_spec, result.x)
    }

    # Compute training metrics with the best config
    best_config_eval = copy.deepcopy(best_config)
    best_config_eval.t_end = float(observed_years[-1]) + 1.0
    best_config_eval.seed = seed if seed is not None else 42
    # Disable stochastic for deterministic eval
    best_config_eval.stochastic.charisma_rate = 0.0
    best_config_eval.stochastic.hybridization_rates[:] = 0.0
    for ideo in best_config_eval.ideologies:
        ideo.mutation_rate = 0.0

    sim_times, sim_shares = _simulate_and_extract(
        best_config_eval, observed_years, sim_ideology_names
    )
    train_metrics = compute_metrics(
        observed_years=observed_years,
        observed_shares=observed_shares,
        simulated_years=sim_times,
        simulated_shares=sim_shares,
        ideology_names=sim_ideology_names,
        metadata={"split": "train", "n_evaluations": result.nfev},
    )

    logger.info(
        "Fitting complete: MSE=%.6f, R2=%.4f (%d evaluations)",
        train_metrics.mse,
        train_metrics.r_squared,
        result.nfev,
    )

    return FitResult(
        best_config=best_config,
        train_metrics=train_metrics,
        parameter_values=param_values,
        n_evaluations=int(result.nfev),
        metadata={
            "optimizer": "differential_evolution",
            "max_iterations": max_iterations,
            "population_size": population_size,
            "converged": bool(result.success),
            "optimizer_message": str(result.message),
        },
    )
