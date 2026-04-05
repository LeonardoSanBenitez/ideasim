#!/usr/bin/env python3
"""Fit simulation parameters against ParlGov European decades dataset.

This script runs the evaluation framework in single-split mode (70/30
train/test) against the ParlGov decade-aggregated data. It saves the
fitted config and evaluation results to output/parlgov_fit/.

Usage:
    python scripts/fit_parlgov.py [--maxiter N] [--popsize N]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ideasim.defaults import make_default_config
from ideasim.evaluation.dataset import load_dataset
from ideasim.evaluation.runner import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit params against ParlGov data")
    parser.add_argument("--maxiter", type=int, default=30, help="DE max iterations")
    parser.add_argument(
        "--popsize", type=int, default=10, help="DE population size multiplier"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--tolerance", type=float, default=1e-5, help="Convergence tolerance"
    )
    parser.add_argument(
        "--train-fraction", type=float, default=0.7, help="Train fraction"
    )
    parser.add_argument(
        "-o", "--output-dir", default="output/parlgov_fit", help="Output directory"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Load dataset
    dataset_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "datasets"
        / "parlgov_europe_decades.json"
    )
    logger.info("Loading dataset from %s", dataset_path)
    dataset = load_dataset(str(dataset_path))
    logger.info(
        "Dataset: %s, %d observations, years %.0f-%.0f, ideologies: %s",
        dataset.name,
        len(dataset.observations),
        dataset.year_range[0],
        dataset.year_range[1],
        ", ".join(dataset.ideology_names),
    )

    # Load default config
    base_config = make_default_config()
    logger.info(
        "Base config: %d ideologies, t=[%.0f, %.0f], dt=%.3f",
        len(base_config.ideologies),
        base_config.t_start,
        base_config.t_end,
        base_config.dt,
    )

    # Estimate runtime
    n_params = 7 * len(base_config.ideologies) + 1  # 7 per ideology + rho
    pop_total = args.popsize * n_params
    est_evals = (args.maxiter + 1) * pop_total
    est_seconds = est_evals * 0.35
    logger.info(
        "Estimated: %d params, pop=%d, ~%d evals, ~%.0f min",
        n_params,
        pop_total,
        est_evals,
        est_seconds / 60,
    )

    # Run evaluation
    t0 = time.time()
    output_dir = Path(args.output_dir)
    eval_result = run_evaluation(
        base_config=base_config,
        dataset=dataset,
        n_folds=1,
        train_fraction=args.train_fraction,
        output_dir=str(output_dir),
        max_iterations=args.maxiter,
        population_size=args.popsize,
        seed=args.seed,
        tolerance=args.tolerance,
        polish=False,
    )
    elapsed = time.time() - t0

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  PARAMETER FITTING RESULTS ({elapsed:.1f}s)")
    print(f"{'=' * 60}")

    fold = eval_result.folds[0]

    print(f"\n  Train period: {fold.split.train_start:.0f}-{fold.split.train_end:.0f}")
    print(f"  Val period:   {fold.split.val_start:.0f}-{fold.split.val_end:.0f}")
    print(f"\n  Train MSE:  {fold.fit_result.train_metrics.mse:.6f}")
    print(f"  Train RMSE: {fold.fit_result.train_metrics.rmse:.6f}")
    print(f"  Train R^2:  {fold.fit_result.train_metrics.r_squared:.4f}")
    print(f"\n  Val MSE:    {fold.val_metrics.mse:.6f}")
    print(f"  Val RMSE:   {fold.val_metrics.rmse:.6f}")
    print(f"  Val R^2:    {fold.val_metrics.r_squared:.4f}")

    print("\n  Per-ideology train MSE:")
    for name, mse in fold.fit_result.train_metrics.per_ideology_mse.items():
        print(f"    {name}: {mse:.6f}")

    print("\n  Per-ideology val MSE:")
    for name, mse in fold.val_metrics.per_ideology_mse.items():
        print(f"    {name}: {mse:.6f}")

    print("\n  Fitted parameters:")
    for name, val in sorted(fold.fit_result.parameter_values.items()):
        print(f"    {name}: {val:.6f}")

    print(f"\n  Optimizer: {fold.fit_result.n_evaluations} evaluations")
    print(f"  Converged: {fold.fit_result.metadata.get('converged', 'N/A')}")
    print(f"\n  Results saved to: {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
