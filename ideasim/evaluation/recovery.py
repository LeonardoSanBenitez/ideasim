"""Parameter recovery: can the fitter find parameters that are known?

A recovery check simulates data from a chosen ("true") parameter vector,
fits the model to that data, and compares the fitted values with the truth.
If a parameter does not come back on synthetic data at a dataset's
resolution, a value fitted to the real dataset says little about the world.

Typical use::

    spec = select_parameters(build_parameter_spec(base), free)
    truth = draw_truth(base, spec, np.random.default_rng(0))
    data = synthetic_dataset(truth, years, noise_sd=0.05, seed=1)
    fit = fit_parameters(base, data, free=free)
    rows = compare_to_truth(fit, truth)

Generate the data with the settings the fit will simulate with (the same
``t_start`` and ``dt``). Otherwise the fit also has to absorb a difference
in integration or epoch, and a recovery failure no longer isolates
identifiability.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ideasim.evaluation.dataset import ObservedDataPoint, ObservedDataset
from ideasim.evaluation.fitting import (
    FitResult,
    FittableParameter,
    _extract_parameters,
    _inject_parameters,
    _simulate_and_extract,
    build_parameter_spec,
    select_parameters,
)
from ideasim.types import SimulationConfig

BOUND_TOLERANCE = 1e-3
"""Fraction of a parameter's bound width within which it counts as pinned."""


def deterministic_copy(config: SimulationConfig, seed: int = 42) -> SimulationConfig:
    """Return a copy with stochastic events and mutation switched off.

    This is the same switch-off ``fit_parameters`` applies to every candidate,
    so data generated from the copy is data the fitter can reproduce exactly.
    """
    out = copy.deepcopy(config)
    out.seed = seed
    out.stochastic.charisma_rate = 0.0
    out.stochastic.hybridization_rates[:] = 0.0
    for ideo in out.ideologies:
        ideo.mutation_rate = 0.0
    return out


def draw_truth(
    base_config: SimulationConfig,
    params_spec: Sequence[FittableParameter],
    rng: np.random.Generator,
    *,
    margin: float = 0.0,
) -> SimulationConfig:
    """Draw each parameter uniformly within its fitting bounds.

    Args:
        base_config: Provides every value not in ``params_spec``.
        params_spec: The parameters to draw (as passed to the fitter).
        rng: Source of randomness; the caller owns the seed.
        margin: Fraction of each bound width excluded at both ends, in
            [0, 0.5). A truth drawn at a bound cannot be told apart from a
            fit pinned there, so a campaign may want a margin.

    Returns:
        A new config. Initial shares, if drawn, are renormalized to the
        simplex exactly as the fitter does, so read the true values back
        with ``compare_to_truth`` rather than from the draw.
    """
    if not 0.0 <= margin < 0.5:
        raise ValueError(f"margin must be in [0, 0.5), got {margin}")
    if not params_spec:
        raise ValueError("params_spec is empty; there is nothing to draw.")
    low = np.array([p.low for p in params_spec], dtype=np.float64)
    high = np.array([p.high for p in params_spec], dtype=np.float64)
    width = high - low
    values = rng.uniform(low + margin * width, high - margin * width)
    return _inject_parameters(base_config, list(params_spec), values)


def synthetic_dataset(
    config: SimulationConfig,
    years: Sequence[float],
    ideology_names: Sequence[str] | None = None,
    *,
    noise_sd: float = 0.0,
    seed: int | None = None,
    name: str = "synthetic",
) -> ObservedDataset:
    """Simulate ``config`` and observe its shares at ``years``.

    The simulation runs deterministically (``deterministic_copy``) up to one
    year after the last requested year, and shares are interpolated at the
    requested years the way ``compute_metrics`` interpolates them. With
    ``noise_sd`` > 0, independent Gaussian noise is added to every share,
    then clipped to [0, 1]; shares are not renormalized, as observed shares
    of a subset of party families need not sum to one.

    Args:
        config: The true configuration.
        years: Observation years, strictly increasing.
        ideology_names: Ideologies to observe (default: all in ``config``).
        noise_sd: Standard deviation of the additive noise, >= 0.
        seed: Seed for the noise (and the simulation's own seed).
        name: Dataset name.
    """
    obs_years = np.asarray(list(years), dtype=np.float64)
    if obs_years.size == 0:
        raise ValueError("years is empty.")
    if np.any(np.diff(obs_years) <= 0):
        raise ValueError("years must be strictly increasing.")
    if noise_sd < 0:
        raise ValueError(f"noise_sd must be >= 0, got {noise_sd}")
    all_names = config.ideology_names()
    names = list(ideology_names) if ideology_names is not None else all_names
    missing = [n for n in names if n not in all_names]
    if missing:
        raise ValueError(f"ideologies {missing} are not in the config {all_names}")
    if obs_years[0] < config.t_start:
        raise ValueError(
            f"first year {obs_years[0]} is before the config's t_start "
            f"{config.t_start}; the simulation cannot observe it."
        )

    sim_config = deterministic_copy(config, seed if seed is not None else 42)
    sim_config.t_end = float(obs_years[-1]) + 1.0
    sim_times, sim_shares = _simulate_and_extract(sim_config, obs_years, names)

    shares = np.column_stack(
        [np.interp(obs_years, sim_times, sim_shares[:, i]) for i in range(len(names))]
    )
    if noise_sd > 0:
        rng = np.random.default_rng(seed)
        shares = np.clip(shares + rng.normal(0.0, noise_sd, shares.shape), 0.0, 1.0)

    observations = [
        ObservedDataPoint(year=float(y), ideology=n, share=float(shares[t, i]))
        for t, y in enumerate(obs_years)
        for i, n in enumerate(names)
    ]
    metadata: dict[str, Any] = {
        "synthetic": True,
        "noise_sd": noise_sd,
        "seed": seed,
        "t_start": config.t_start,
        "dt": config.dt,
    }
    return ObservedDataset(
        name=name,
        description="Simulated from a known configuration (parameter recovery).",
        ideology_names=names,
        observations=observations,
        metadata=metadata,
    )


@dataclass(frozen=True)
class ParameterRecovery:
    """One parameter's true and fitted value.

    Attributes:
        name: Parameter name, as in ``build_parameter_spec``.
        true: Value in the true configuration.
        fitted: Value in the fitted configuration (after any
            renormalization of initial shares, i.e. the value simulated).
        low: Lower fitting bound.
        high: Upper fitting bound.
    """

    name: str
    true: float
    fitted: float
    low: float
    high: float

    @property
    def error(self) -> float:
        """Fitted minus true."""
        return self.fitted - self.true

    @property
    def scaled_error(self) -> float:
        """Error as a fraction of the bound width, so parameters compare."""
        return self.error / (self.high - self.low)

    @property
    def at_bound(self) -> bool:
        """Whether the fit ended within BOUND_TOLERANCE of a bound."""
        tol = BOUND_TOLERANCE * (self.high - self.low)
        return self.fitted - self.low <= tol or self.high - self.fitted <= tol

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form, including the derived fields."""
        return {
            "name": self.name,
            "true": self.true,
            "fitted": self.fitted,
            "low": self.low,
            "high": self.high,
            "error": self.error,
            "scaled_error": self.scaled_error,
            "at_bound": self.at_bound,
        }


def compare_to_truth(
    fit: FitResult, truth: SimulationConfig
) -> list[ParameterRecovery]:
    """Compare every fitted parameter of ``fit`` with its value in ``truth``.

    The parameters are those the fit optimized (the keys of
    ``fit.parameter_values``), with the bounds and ``include_*`` choices
    recorded in ``fit.metadata``. Both values are read from configurations,
    so initial shares are compared after renormalization on both sides.

    Raises:
        ValueError: if ``fit.metadata`` lacks the ``include_*`` flags (a
            FitResult not produced by ``fit_parameters``), or names a
            parameter ``truth`` does not have.
    """
    try:
        shares = bool(fit.metadata["include_initial_shares"])
        latent = bool(fit.metadata["include_initial_latent"])
    except KeyError as exc:
        raise ValueError(
            f"fit.metadata has no {exc}; compare_to_truth needs a FitResult "
            "from fit_parameters to know which parameters were fitted."
        ) from exc
    spec = select_parameters(
        build_parameter_spec(
            truth, include_initial_shares=shares, include_initial_latent=latent
        ),
        list(fit.parameter_values),
    )
    true_values = _extract_parameters(truth, spec)
    fitted_values = _extract_parameters(fit.best_config, spec)
    return [
        ParameterRecovery(
            name=p.name, true=float(t), fitted=float(f), low=p.low, high=p.high
        )
        for p, t, f in zip(spec, true_values, fitted_values, strict=True)
    ]
