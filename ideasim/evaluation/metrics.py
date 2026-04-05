"""Evaluation metrics for comparing simulated vs observed ideology shares.

All metrics handle NaN values (missing ideology observations) gracefully
by only computing over non-NaN entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics.

    Attributes:
        mse: Mean Squared Error over all (year, ideology) pairs.
        rmse: Root Mean Squared Error.
        mae: Mean Absolute Error.
        r_squared: R-squared (coefficient of determination).
        per_ideology_mse: MSE broken down by ideology.
        per_ideology_mae: MAE broken down by ideology.
        per_ideology_r_squared: R-squared broken down by ideology.
        n_observations: Number of non-NaN observation pairs used.
        max_absolute_error: Worst single-point error.
        metadata: Additional info (e.g., which split, fold index).
    """

    mse: float
    rmse: float
    mae: float
    r_squared: float
    per_ideology_mse: dict[str, float]
    per_ideology_mae: dict[str, float]
    per_ideology_r_squared: dict[str, float]
    n_observations: int
    max_absolute_error: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "mse": round(self.mse, 8),
            "rmse": round(self.rmse, 8),
            "mae": round(self.mae, 8),
            "r_squared": round(self.r_squared, 8),
            "per_ideology_mse": {
                k: round(v, 8) for k, v in self.per_ideology_mse.items()
            },
            "per_ideology_mae": {
                k: round(v, 8) for k, v in self.per_ideology_mae.items()
            },
            "per_ideology_r_squared": {
                k: round(v, 8) for k, v in self.per_ideology_r_squared.items()
            },
            "n_observations": self.n_observations,
            "max_absolute_error": round(self.max_absolute_error, 8),
            "metadata": self.metadata,
        }


def _r_squared(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Compute R-squared, handling edge cases."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-15:
        # All observations are identical; R^2 is undefined.
        # Return 1.0 if predictions are also constant and match, else 0.0.
        if ss_res < 1e-15:
            return 1.0
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_metrics(
    observed_years: NDArray[np.float64],
    observed_shares: NDArray[np.float64],
    simulated_years: NDArray[np.float64],
    simulated_shares: NDArray[np.float64],
    ideology_names: list[str],
    metadata: dict[str, Any] | None = None,
) -> EvaluationMetrics:
    """Compute evaluation metrics between observed and simulated shares.

    The simulation may produce shares at different time points than the
    observations. This function interpolates simulated shares to match
    observed years.

    Args:
        observed_years: shape (T_obs,) - years with observations.
        observed_shares: shape (T_obs, N) - observed shares. NaN for missing.
        simulated_years: shape (T_sim,) - years from simulation output.
        simulated_shares: shape (T_sim, N) - simulated shares.
        ideology_names: names for each ideology column.
        metadata: optional metadata to attach to results.

    Returns:
        EvaluationMetrics with all computed metrics.
    """
    n_ideo = len(ideology_names)

    # Interpolate simulated shares to observed years
    interp_shares = np.full_like(observed_shares, np.nan)
    for i in range(n_ideo):
        sim_col = simulated_shares[:, i]
        # Only interpolate where simulated data is valid
        valid_sim = ~np.isnan(sim_col)
        if np.sum(valid_sim) < 1:
            continue
        interp_shares[:, i] = np.interp(
            observed_years,
            simulated_years[valid_sim],
            sim_col[valid_sim],
        )

    # Build mask: only where both observed and interpolated are non-NaN
    mask = ~np.isnan(observed_shares) & ~np.isnan(interp_shares)
    n_obs = int(np.sum(mask))

    if n_obs == 0:
        return EvaluationMetrics(
            mse=float("nan"),
            rmse=float("nan"),
            mae=float("nan"),
            r_squared=float("nan"),
            per_ideology_mse={},
            per_ideology_mae={},
            per_ideology_r_squared={},
            n_observations=0,
            max_absolute_error=float("nan"),
            metadata=metadata or {},
        )

    obs_flat = observed_shares[mask]
    pred_flat = interp_shares[mask]
    errors = obs_flat - pred_flat

    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    r2 = _r_squared(obs_flat, pred_flat)
    max_err = float(np.max(np.abs(errors)))

    # Per-ideology metrics
    per_mse: dict[str, float] = {}
    per_mae: dict[str, float] = {}
    per_r2: dict[str, float] = {}

    for i, name in enumerate(ideology_names):
        col_mask = mask[:, i]
        if np.sum(col_mask) == 0:
            continue
        obs_col = observed_shares[col_mask, i]
        pred_col = interp_shares[col_mask, i]
        col_errors = obs_col - pred_col
        per_mse[name] = float(np.mean(col_errors**2))
        per_mae[name] = float(np.mean(np.abs(col_errors)))
        per_r2[name] = _r_squared(obs_col, pred_col)

    return EvaluationMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        r_squared=r2,
        per_ideology_mse=per_mse,
        per_ideology_mae=per_mae,
        per_ideology_r_squared=per_r2,
        n_observations=n_obs,
        max_absolute_error=max_err,
        metadata=metadata or {},
    )
