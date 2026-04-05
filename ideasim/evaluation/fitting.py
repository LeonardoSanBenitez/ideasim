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
from scipy.optimize import Bounds, OptimizeResult, differential_evolution

from ideasim.engine import simulate
from ideasim.evaluation.dataset import ObservedDataset
from ideasim.evaluation.metrics import EvaluationMetrics, compute_metrics
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
    *,
    include_initial_shares: bool = False,
    include_initial_latent: bool = False,
) -> list[FittableParameter]:
    """Build the list of fittable parameters for a given config.

    Returns one FittableParameter per tunable scalar in the config.

    Args:
        config: The simulation config (provides structure and initial values).
        include_initial_shares: If True, also fit per-ideology initial shares
            (constrained to [0.01, 0.95]).
        include_initial_latent: If True, also fit per-ideology initial latent
            sympathy values (constrained to [0.01, 1.0]).
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

    if include_initial_shares:
        for i, ideo in enumerate(config.ideologies):
            params.append(
                FittableParameter(
                    name=f"initial_share.{ideo.name}",
                    low=0.01,
                    high=0.95,
                    _ideology_idx=i,
                    _param_name="__initial_share__",
                )
            )

    if include_initial_latent:
        for i, ideo in enumerate(config.ideologies):
            params.append(
                FittableParameter(
                    name=f"initial_latent.{ideo.name}",
                    low=0.01,
                    high=1.0,
                    _ideology_idx=i,
                    _param_name="__initial_latent__",
                )
            )

    return params


def _inject_parameters(
    config: SimulationConfig,
    params_spec: list[FittableParameter],
    values: NDArray[np.float64],
) -> SimulationConfig:
    """Create a new config with the given parameter values injected.

    Handles standard ideology params, global params, and special
    __initial_share__ / __initial_latent__ synthetic params.
    When initial shares are injected, they are renormalized to sum to 1.
    """
    new_config = copy.deepcopy(config)

    shares_modified = False
    for param, value in zip(params_spec, values, strict=True):
        if param._param_name == "__initial_share__":
            if param._ideology_idx is not None:
                new_config.initial_shares[param._ideology_idx] = float(value)
                shares_modified = True
        elif param._param_name == "__initial_latent__":
            if param._ideology_idx is not None:
                new_config.initial_latent[param._ideology_idx] = float(value)
        elif param._ideology_idx is not None and param._param_name is not None:
            setattr(
                new_config.ideologies[param._ideology_idx],
                param._param_name,
                float(value),
            )
        elif param._param_name is not None:
            setattr(new_config, param._param_name, float(value))

    # Renormalize initial shares to the simplex
    if shares_modified:
        total = float(np.sum(new_config.initial_shares))
        if total > 0:
            new_config.initial_shares = new_config.initial_shares / total

    return new_config


def _extract_parameters(
    config: SimulationConfig,
    params_spec: list[FittableParameter],
) -> NDArray[np.float64]:
    """Extract current parameter values from a config."""
    values = np.zeros(len(params_spec), dtype=np.float64)

    for i, param in enumerate(params_spec):
        if param._param_name == "__initial_share__":
            if param._ideology_idx is not None:
                values[i] = config.initial_shares[param._ideology_idx]
        elif param._param_name == "__initial_latent__":
            if param._ideology_idx is not None:
                values[i] = config.initial_latent[param._ideology_idx]
        elif param._ideology_idx is not None and param._param_name is not None:
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
    polish: bool = False,
    include_initial_shares: bool = False,
    include_initial_latent: bool = False,
    align_t_start: bool = False,
    fit_dt: float | None = None,
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
        polish: Whether to run L-BFGS-B local optimization after DE
            (default: False, since it can be slow with many parameters).
        include_initial_shares: If True, also optimize initial ideology
            shares (renormalized to the simplex after each evaluation).
        include_initial_latent: If True, also optimize initial latent
            sympathy values.
        align_t_start: If True, set t_start to 10 years before the
            first observation year during fitting. This greatly speeds
            up fitting when observations start much later than the
            default t_start (e.g., 1900 vs 1800). Note: exogenous
            shocks before the aligned t_start will be skipped.
        fit_dt: If provided, use this dt during fitting instead of the
            config's dt. Larger dt (e.g., 0.5) speeds up fitting at
            the cost of integration accuracy. The final best_config
            retains the original dt.

    Returns:
        FitResult with optimized config and training metrics.
    """
    sim_ideology_names = base_config.ideology_names()
    observed_years, observed_shares = train_dataset.as_matrix(sim_ideology_names)

    if len(observed_years) == 0:
        raise ValueError("Training dataset has no observations.")

    params_spec = build_parameter_spec(
        base_config,
        include_initial_shares=include_initial_shares,
        include_initial_latent=include_initial_latent,
    )
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

    # Optionally align t_start to first observation for speed
    if align_t_start:
        aligned_start = float(observed_years[0]) - 10.0
        if aligned_start > fit_config.t_start:
            logger.info(
                "Aligning t_start from %.0f to %.0f (first obs at %.0f)",
                fit_config.t_start,
                aligned_start,
                observed_years[0],
            )
            fit_config.t_start = aligned_start

    # Optionally use coarser dt during fitting for speed
    original_dt = fit_config.dt
    if fit_dt is not None:
        logger.info("Using fit_dt=%.3f (original dt=%.3f)", fit_dt, original_dt)
        fit_config.dt = fit_dt

    _gen_count = [0]

    def _progress_callback(
        xk: NDArray[np.float64],
        convergence: float | np.floating[Any] | np.integer[Any] | np.bool_[bool] = 0.0,
    ) -> None:
        """Log progress after each DE generation."""
        _gen_count[0] += 1
        logger.info(
            "  DE generation %d/%d, convergence=%.6f",
            _gen_count[0],
            max_iterations,
            float(convergence),
        )

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
        polish=polish,
        callback=_progress_callback,
    )

    # Build the best config
    best_config = _inject_parameters(base_config, params_spec, result.x)
    param_values = {
        p.name: float(v) for p, v in zip(params_spec, result.x, strict=True)
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
            "polish": polish,
            "optimizer_message": str(result.message),
            "include_initial_shares": include_initial_shares,
            "include_initial_latent": include_initial_latent,
            "align_t_start": align_t_start,
            "fit_dt": fit_dt,
        },
    )
