"""Evaluation runner: orchestrates fitting, validation, and reporting.

Supports both single train/test split and temporal k-fold CV.
All results are saved to JSON for reproducibility.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ideasim.evaluation.dataset import ObservedDataset
from ideasim.evaluation.fitting import FitResult, fit_parameters
from ideasim.evaluation.metrics import EvaluationMetrics, compute_metrics
from ideasim.evaluation.splits import (
    TemporalSplit,
    temporal_kfold,
    temporal_train_test_split,
)
from ideasim.io import save_config
from ideasim.types import SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Result for a single fold in cross-validation."""

    fold_index: int
    split: TemporalSplit
    fit_result: FitResult
    val_metrics: EvaluationMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "split": {
                "train_start": self.split.train_start,
                "train_end": self.split.train_end,
                "val_end": self.split.val_end,
            },
            "fit": self.fit_result.to_dict(),
            "val_metrics": self.val_metrics.to_dict(),
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result across all folds.

    Attributes:
        dataset_name: Name of the dataset used.
        folds: Per-fold results.
        aggregate_val_metrics: Aggregated validation metrics across folds.
        elapsed_seconds: Total wall-clock time.
        metadata: Additional info.
    """

    dataset_name: str
    folds: list[FoldResult]
    aggregate_val_metrics: dict[str, Any]
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "n_folds": len(self.folds),
            "aggregate_val_metrics": self.aggregate_val_metrics,
            "folds": [f.to_dict() for f in self.folds],
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        """Save evaluation result to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def _evaluate_fold(
    base_config: SimulationConfig,
    dataset: ObservedDataset,
    split: TemporalSplit,
    fit_kwargs: dict[str, Any],
) -> FoldResult:
    """Run fitting and validation for a single fold."""
    train_ds, val_ds = split.split_dataset(dataset)

    logger.info(
        "Fold %d: train [%.0f-%.0f] (%d years), val (%.0f-%.0f] (%d years)",
        split.fold_index,
        split.train_start,
        split.train_end,
        len(train_ds.years),
        split.val_start,
        split.val_end,
        len(val_ds.years),
    )

    # Adjust config time range to cover training period
    fit_config = copy.deepcopy(base_config)

    # Fit parameters on training data
    fit_result = fit_parameters(
        base_config=fit_config,
        train_dataset=train_ds,
        **fit_kwargs,
    )

    # Evaluate on validation data using the fitted config
    sim_ideology_names = base_config.ideology_names()
    val_years, val_shares = val_ds.as_matrix(sim_ideology_names)

    if len(val_years) == 0:
        logger.warning("Fold %d: no validation observations.", split.fold_index)
        val_metrics = EvaluationMetrics(
            mse=float("nan"),
            rmse=float("nan"),
            mae=float("nan"),
            r_squared=float("nan"),
            per_ideology_mse={},
            per_ideology_mae={},
            per_ideology_r_squared={},
            n_observations=0,
            max_absolute_error=float("nan"),
            metadata={"split": "val", "fold_index": split.fold_index},
        )
    else:
        # Run simulation with fitted params through validation period
        eval_config = copy.deepcopy(fit_result.best_config)
        eval_config.t_end = float(val_years[-1]) + 1.0
        eval_config.seed = fit_kwargs.get("seed", 42)
        # Disable stochastic for deterministic eval
        eval_config.stochastic.charisma_rate = 0.0
        eval_config.stochastic.hybridization_rates[:] = 0.0
        for ideo in eval_config.ideologies:
            ideo.mutation_rate = 0.0

        from ideasim.evaluation.fitting import _simulate_and_extract

        sim_times, sim_shares = _simulate_and_extract(
            eval_config, val_years, sim_ideology_names
        )

        val_metrics = compute_metrics(
            observed_years=val_years,
            observed_shares=val_shares,
            simulated_years=sim_times,
            simulated_shares=sim_shares,
            ideology_names=sim_ideology_names,
            metadata={"split": "val", "fold_index": split.fold_index},
        )

    logger.info(
        "Fold %d: train MSE=%.6f, val MSE=%.6f, val R2=%.4f",
        split.fold_index,
        fit_result.train_metrics.mse,
        val_metrics.mse,
        val_metrics.r_squared,
    )

    return FoldResult(
        fold_index=split.fold_index,
        split=split,
        fit_result=fit_result,
        val_metrics=val_metrics,
    )


def _aggregate_metrics(folds: list[FoldResult]) -> dict[str, Any]:
    """Aggregate validation metrics across folds."""
    val_mses = [f.val_metrics.mse for f in folds if not np.isnan(f.val_metrics.mse)]
    val_maes = [f.val_metrics.mae for f in folds if not np.isnan(f.val_metrics.mae)]
    val_r2s = [
        f.val_metrics.r_squared
        for f in folds
        if not np.isnan(f.val_metrics.r_squared)
    ]

    result: dict[str, Any] = {
        "n_valid_folds": len(val_mses),
    }

    if val_mses:
        result["mse_mean"] = round(float(np.mean(val_mses)), 8)
        result["mse_std"] = round(float(np.std(val_mses)), 8)
        result["rmse_mean"] = round(float(np.sqrt(np.mean(val_mses))), 8)
    if val_maes:
        result["mae_mean"] = round(float(np.mean(val_maes)), 8)
        result["mae_std"] = round(float(np.std(val_maes)), 8)
    if val_r2s:
        result["r_squared_mean"] = round(float(np.mean(val_r2s)), 8)
        result["r_squared_std"] = round(float(np.std(val_r2s)), 8)

    return result


def run_evaluation(
    base_config: SimulationConfig,
    dataset: ObservedDataset,
    n_folds: int = 5,
    train_fraction: float = 0.7,
    output_dir: str | Path | None = None,
    max_iterations: int = 100,
    population_size: int = 15,
    seed: int | None = 42,
    tolerance: float = 1e-6,
) -> EvaluationResult:
    """Run a full evaluation: temporal k-fold CV with fitting and validation.

    If n_folds == 1, does a single temporal train/test split using
    train_fraction. If n_folds > 1, does expanding-window temporal
    k-fold cross-validation.

    Args:
        base_config: The simulation config to use as starting point.
        dataset: Observed ideology share data.
        n_folds: Number of CV folds (1 for single split).
        train_fraction: Fraction of years for training (single split only).
        output_dir: Directory to save results. If None, results are
            not saved to disk.
        max_iterations: Max optimizer iterations per fold.
        population_size: DE population size multiplier.
        seed: Random seed.
        tolerance: Optimizer convergence tolerance.

    Returns:
        EvaluationResult with per-fold and aggregate metrics.
    """
    t0 = time.time()

    # Build splits
    if n_folds == 1:
        split = temporal_train_test_split(dataset, train_fraction)
        splits = [split]
    else:
        splits = temporal_kfold(dataset, n_folds=n_folds)

    fit_kwargs: dict[str, Any] = {
        "max_iterations": max_iterations,
        "population_size": population_size,
        "seed": seed,
        "tolerance": tolerance,
    }

    # Run each fold
    fold_results: list[FoldResult] = []
    for split in splits:
        fold_result = _evaluate_fold(base_config, dataset, split, fit_kwargs)
        fold_results.append(fold_result)

    # Aggregate
    agg = _aggregate_metrics(fold_results)

    elapsed = time.time() - t0
    eval_result = EvaluationResult(
        dataset_name=dataset.name,
        folds=fold_results,
        aggregate_val_metrics=agg,
        elapsed_seconds=elapsed,
        metadata={
            "n_folds": n_folds,
            "train_fraction": train_fraction,
            "max_iterations": max_iterations,
            "population_size": population_size,
            "seed": seed,
            "tolerance": tolerance,
            "ideology_names": base_config.ideology_names(),
        },
    )

    # Save results
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        eval_result.save(out_path / "evaluation_results.json")
        # Also save the best config from the last fold (or single split)
        best_fold = fold_results[-1]
        save_config(
            best_fold.fit_result.best_config,
            out_path / "fitted_config.json",
        )
        logger.info("Results saved to %s", out_path)

    logger.info(
        "Evaluation complete: %d folds, %.1fs, aggregate val MSE=%.6f",
        len(fold_results),
        elapsed,
        agg.get("mse_mean", float("nan")),
    )

    return eval_result
