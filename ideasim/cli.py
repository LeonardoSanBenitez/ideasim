"""CLI entry point for the Ideological Dynamics Simulator."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from ideasim.defaults import make_default_config
from ideasim.engine import simulate
from ideasim.io import load_config, save_config, save_result


def _add_simulate_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    sim_parser = subparsers.add_parser("simulate", help="Run a simulation")
    sim_parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to JSON config file. If omitted, uses default European 1800-2025 scenario.",
    )
    sim_parser.add_argument(
        "-o", "--output",
        default="results.json",
        help="Output path for results JSON (default: results.json)",
    )


def _add_default_config_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    def_parser = subparsers.add_parser("default-config", help="Export the default config to JSON")
    def_parser.add_argument(
        "-o", "--output",
        default="default_config.json",
        help="Output path (default: default_config.json)",
    )


def _add_evaluate_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate simulation against observed data",
    )
    eval_parser.add_argument(
        "dataset",
        help="Path to observed dataset JSON file.",
    )
    eval_parser.add_argument(
        "--config",
        default=None,
        help="Path to base simulation config JSON. If omitted, uses default.",
    )
    eval_parser.add_argument(
        "-o", "--output-dir",
        default="output/evaluation",
        help="Directory for evaluation results (default: output/evaluation)",
    )
    eval_parser.add_argument(
        "--n-folds",
        type=int,
        default=1,
        help="Number of CV folds. 1 = single train/test split (default: 1).",
    )
    eval_parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Fraction of years for training in single-split mode (default: 0.7).",
    )
    eval_parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Max optimizer iterations per fold (default: 100).",
    )
    eval_parser.add_argument(
        "--population-size",
        type=int,
        default=15,
        help="Differential evolution population size multiplier (default: 15).",
    )
    eval_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    eval_parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Optimizer convergence tolerance (default: 1e-6).",
    )
    eval_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )


def _run_simulate(args: argparse.Namespace) -> None:
    if args.config:
        print(f"Loading config from {args.config}...")
        config = load_config(args.config)
    else:
        print("Using default European 1800-2025 scenario...")
        config = make_default_config()

    print(
        f"Simulating {config.n_ideologies} ideologies "
        f"from {config.t_start:.0f} to {config.t_end:.0f} "
        f"(dt={config.dt}, seed={config.seed})..."
    )

    t0 = time.time()
    result = simulate(config)
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.2f}s — {result.metadata['total_steps']} steps, "
          f"{result.metadata['total_events']} stochastic events")

    # Print summary
    print("\n--- Summary ---")
    last = result.history[-1]
    for i, name in enumerate(last.ideology_names):
        if i < len(last.shares):
            share = last.shares[i]
            print(f"  {name}: {share:.1%}")

    hyb_count = sum(1 for e in result.events if e.event_type == "hybridization")
    if hyb_count > 0:
        print(f"\n  Hybrid ideologies emerged: {hyb_count}")
        for ev in result.events:
            if ev.event_type == "hybridization":
                print(f"    {ev.details['child']} ({ev.details['parents'][0]} + "
                      f"{ev.details['parents'][1]}) at year {ev.time:.0f}")

    save_result(result, args.output)
    print(f"\nFull results saved to {args.output}")


def _run_evaluate(args: argparse.Namespace) -> None:
    from ideasim.evaluation.dataset import load_dataset
    from ideasim.evaluation.runner import run_evaluation

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(args.dataset)
    print(f"  Dataset: {dataset.name}")
    print(f"  Years: {dataset.year_range[0]:.0f}-{dataset.year_range[1]:.0f}")
    print(f"  Ideologies: {', '.join(dataset.ideology_names)}")
    print(f"  Observations: {len(dataset.observations)}")

    # Load or create base config
    if args.config:
        print(f"Loading base config from {args.config}...")
        base_config = load_config(args.config)
    else:
        print("Using default simulation config as base...")
        base_config = make_default_config()

    # Run evaluation
    mode = f"{args.n_folds}-fold temporal CV" if args.n_folds > 1 else "single temporal split"
    print(f"\nRunning evaluation ({mode})...")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Population size: {args.population_size}")
    print(f"  Seed: {args.seed}")

    eval_result = run_evaluation(
        base_config=base_config,
        dataset=dataset,
        n_folds=args.n_folds,
        train_fraction=args.train_fraction,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        population_size=args.population_size,
        seed=args.seed,
        tolerance=args.tolerance,
    )

    # Print results
    print(f"\n--- Evaluation Results ({eval_result.elapsed_seconds:.1f}s) ---")
    agg = eval_result.aggregate_val_metrics
    if "mse_mean" in agg:
        print(f"  Validation MSE:  {agg['mse_mean']:.6f} (+/- {agg.get('mse_std', 0):.6f})")
        print(f"  Validation RMSE: {agg['rmse_mean']:.6f}")
    if "mae_mean" in agg:
        print(f"  Validation MAE:  {agg['mae_mean']:.6f} (+/- {agg.get('mae_std', 0):.6f})")
    if "r_squared_mean" in agg:
        print(f"  Validation R^2:  {agg['r_squared_mean']:.4f} (+/- {agg.get('r_squared_std', 0):.4f})")

    for fold in eval_result.folds:
        print(f"\n  Fold {fold.fold_index}: "
              f"train [{fold.split.train_start:.0f}-{fold.split.train_end:.0f}], "
              f"val ({fold.split.train_end:.0f}-{fold.split.val_end:.0f}]")
        print(f"    Train MSE: {fold.fit_result.train_metrics.mse:.6f}")
        print(f"    Val MSE:   {fold.val_metrics.mse:.6f}")
        print(f"    Val R^2:   {fold.val_metrics.r_squared:.4f}")

    print(f"\nResults saved to {args.output_dir}/")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ideasim",
        description="Ideological Dynamics Simulator — models competition between philosophical currents",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_simulate_parser(subparsers)
    _add_default_config_parser(subparsers)
    _add_evaluate_parser(subparsers)

    args = parser.parse_args(argv)

    if args.command == "default-config":
        config = make_default_config()
        save_config(config, args.output)
        print(f"Default config written to {args.output}")
        return

    if args.command == "simulate":
        _run_simulate(args)

    if args.command == "evaluate":
        _run_evaluate(args)


if __name__ == "__main__":
    main()
