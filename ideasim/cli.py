"""CLI entry point for the Ideological Dynamics Simulator."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np

from ideasim.defaults import make_default_config
from ideasim.engine import simulate
from ideasim.health import check_health
from ideasim.io import load_config, save_config, save_result
from ideasim.sensitivity import run_sensitivity
from ideasim.validation import validate_config


def _add_simulate_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    sim_parser = subparsers.add_parser("simulate", help="Run a simulation")
    sim_parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to JSON config file. If omitted, uses default European 1800-2025 scenario.",
    )
    sim_parser.add_argument(
        "-o",
        "--output",
        default="results.json",
        help="Output path for results JSON (default: results.json)",
    )
    sim_parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip config validation before running.",
    )
    sim_parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip post-simulation health checks.",
    )


def _add_default_config_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    def_parser = subparsers.add_parser(
        "default-config", help="Export the default config to JSON"
    )
    def_parser.add_argument(
        "-o",
        "--output",
        default="default_config.json",
        help="Output path (default: default_config.json)",
    )


def _add_evaluate_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
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
        "-o",
        "--output-dir",
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
        "--polish",
        action="store_true",
        help="Run L-BFGS-B local optimization after DE (default: off).",
    )
    eval_parser.add_argument(
        "--fit-initial-shares",
        action="store_true",
        help="Also optimize initial ideology shares (default: off).",
    )
    eval_parser.add_argument(
        "--fit-initial-latent",
        action="store_true",
        help="Also optimize initial latent sympathy values (default: off).",
    )
    eval_parser.add_argument(
        "--align-t-start",
        action="store_true",
        help="Align t_start to near first observation year for faster fitting.",
    )
    eval_parser.add_argument(
        "--fit-dt",
        type=float,
        default=None,
        help="Use coarser dt during fitting for speed (e.g., 0.5).",
    )
    eval_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )


def _add_validate_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    val_parser = subparsers.add_parser(
        "validate",
        help="Validate a simulation config without running it",
    )
    val_parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to JSON config file. If omitted, validates the default config.",
    )


def _add_check_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    check_parser = subparsers.add_parser(
        "check",
        help="Run simulation and check health of results",
    )
    check_parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to JSON config file. If omitted, uses default.",
    )
    check_parser.add_argument(
        "--json",
        action="store_true",
        help="Output health report as JSON.",
    )


def _add_ingest_parlgov_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    ingest_parser = subparsers.add_parser(
        "ingest-parlgov",
        help="Download ParlGov data and produce an ObservedDataset JSON",
    )
    ingest_parser.add_argument(
        "-o",
        "--output",
        default="parlgov_dataset.json",
        help="Output path for the dataset JSON (default: parlgov_dataset.json)",
    )
    ingest_parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache downloaded CSV files. If omitted, uses data/parlgov_cache/.",
    )
    ingest_parser.add_argument(
        "--election-csv",
        default=None,
        help="Path to an already-downloaded view_election.csv (skips download).",
    )
    ingest_parser.add_argument(
        "--party-csv",
        default=None,
        help="Path to an already-downloaded view_party.csv (skips download).",
    )
    ingest_parser.add_argument(
        "--per-year",
        action="store_true",
        help="Aggregate per election year instead of per decade.",
    )
    ingest_parser.add_argument(
        "--name",
        default="parlgov_europe",
        help="Dataset name (default: parlgov_europe).",
    )


def _add_sensitivity_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    sens_parser = subparsers.add_parser(
        "sensitivity",
        help="Run sensitivity analysis (OAT and/or LHS)",
    )
    sens_parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to JSON config file. If omitted, uses default.",
    )
    sens_parser.add_argument(
        "-o",
        "--output",
        default="sensitivity_results.json",
        help="Output path for results JSON (default: sensitivity_results.json)",
    )
    sens_parser.add_argument(
        "--oat-steps",
        type=int,
        default=10,
        help="Number of values per parameter in OAT sweep (default: 10).",
    )
    sens_parser.add_argument(
        "--lhs-samples",
        type=int,
        default=50,
        help="Number of LHS samples (default: 50).",
    )
    sens_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for LHS sampling (default: 42).",
    )
    sens_parser.add_argument(
        "--no-oat",
        action="store_true",
        help="Skip OAT analysis.",
    )
    sens_parser.add_argument(
        "--no-lhs",
        action="store_true",
        help="Skip LHS analysis.",
    )
    sens_parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Enable stochastic events (default: deterministic for reproducibility).",
    )
    sens_parser.add_argument(
        "--json",
        action="store_true",
        help="Output full results as JSON to stdout instead of human-readable summary.",
    )


def _run_simulate(args: argparse.Namespace) -> None:
    if args.config:
        print(f"Loading config from {args.config}...")
        config = load_config(args.config)
    else:
        print("Using default European 1800-2025 scenario...")
        config = make_default_config()

    # Validate config
    if not args.skip_validation:
        validation = validate_config(config)
        if not validation.is_valid:
            print("\nConfig validation FAILED:")
            print(validation.summary())
            sys.exit(1)
        if validation.warnings:
            print(
                f"\nConfig validation passed with {len(validation.warnings)} warning(s):"
            )
            for warn in validation.warnings:
                print(f"  {warn}")

    print(
        f"Simulating {config.n_ideologies} ideologies "
        f"from {config.t_start:.0f} to {config.t_end:.0f} "
        f"(dt={config.dt}, seed={config.seed})..."
    )

    t0 = time.time()
    result = simulate(config)
    elapsed = time.time() - t0

    print(
        f"Done in {elapsed:.2f}s — {result.metadata['total_steps']} steps, "
        f"{result.metadata['total_events']} stochastic events"
    )

    # Health check
    if not args.skip_health_check:
        health = check_health(result)
        if not health.all_passed:
            print("\n--- Health Check Issues ---")
            print(health.summary())
        else:
            print(f"\nHealth check: all {len(health.checks)} checks passed.")

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
                print(
                    f"    {ev.details['child']} ({ev.details['parents'][0]} + "
                    f"{ev.details['parents'][1]}) at year {ev.time:.0f}"
                )

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

    # Validate config
    validation = validate_config(base_config)
    if not validation.is_valid:
        print("\nConfig validation FAILED:")
        print(validation.summary())
        sys.exit(1)

    # Run evaluation
    mode = (
        f"{args.n_folds}-fold temporal CV"
        if args.n_folds > 1
        else "single temporal split"
    )
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
        polish=args.polish,
        include_initial_shares=args.fit_initial_shares,
        include_initial_latent=args.fit_initial_latent,
        align_t_start=args.align_t_start,
        fit_dt=args.fit_dt,
    )

    # Print results
    print(f"\n--- Evaluation Results ({eval_result.elapsed_seconds:.1f}s) ---")
    agg = eval_result.aggregate_val_metrics
    if "mse_mean" in agg:
        print(
            f"  Validation MSE:  {agg['mse_mean']:.6f} (+/- {agg.get('mse_std', 0):.6f})"
        )
        print(f"  Validation RMSE: {agg['rmse_mean']:.6f}")
    if "mae_mean" in agg:
        print(
            f"  Validation MAE:  {agg['mae_mean']:.6f} (+/- {agg.get('mae_std', 0):.6f})"
        )
    if "r_squared_mean" in agg:
        print(
            f"  Validation R^2:  {agg['r_squared_mean']:.4f} (+/- {agg.get('r_squared_std', 0):.4f})"
        )

    for fold in eval_result.folds:
        print(
            f"\n  Fold {fold.fold_index}: "
            f"train [{fold.split.train_start:.0f}-{fold.split.train_end:.0f}], "
            f"val ({fold.split.train_end:.0f}-{fold.split.val_end:.0f}]"
        )
        print(f"    Train MSE: {fold.fit_result.train_metrics.mse:.6f}")
        print(f"    Val MSE:   {fold.val_metrics.mse:.6f}")
        print(f"    Val R^2:   {fold.val_metrics.r_squared:.4f}")

    print(f"\nResults saved to {args.output_dir}/")


def _run_validate(args: argparse.Namespace) -> None:
    if args.config:
        print(f"Loading config from {args.config}...")
        config = load_config(args.config)
    else:
        print("Validating default config...")
        config = make_default_config()

    validation = validate_config(config)
    print(validation.summary())
    if not validation.is_valid:
        sys.exit(1)


def _run_check(args: argparse.Namespace) -> None:
    if args.config:
        print(f"Loading config from {args.config}...")
        config = load_config(args.config)
    else:
        print("Using default config...")
        config = make_default_config()

    # Validate first
    validation = validate_config(config)
    if not validation.is_valid:
        print("Config validation FAILED:")
        print(validation.summary())
        sys.exit(1)

    # Run simulation
    print(f"Running simulation ({config.t_start:.0f}-{config.t_end:.0f})...")
    t0 = time.time()
    result = simulate(config)
    elapsed = time.time() - t0
    print(f"Simulation completed in {elapsed:.2f}s.")

    # Health check
    health = check_health(result)

    if args.json:
        report_data = health.to_dict()
        report_data["simulation_time_seconds"] = round(elapsed, 2)
        report_data["total_steps"] = result.metadata["total_steps"]
        report_data["total_events"] = result.metadata["total_events"]
        print(json.dumps(report_data, indent=2))
    else:
        print("\n--- Health Report ---")
        print(health.summary())

        # Also print basic metrics
        last = result.history[-1]
        print("\n--- Final State ---")
        for i, name in enumerate(last.ideology_names):
            if i < len(last.shares):
                print(f"  {name}: {last.shares[i]:.4f}")

        entropy = result.entropy_timeseries()
        if len(entropy) > 0:
            print(
                f"\n  Entropy: {float(entropy[-1]):.4f} "
                f"(range [{float(np.min(entropy)):.4f}, {float(np.max(entropy)):.4f}])"
            )

    if not health.all_passed:
        sys.exit(1)


def _run_ingest_parlgov(args: argparse.Namespace) -> None:
    from pathlib import Path

    from ideasim.evaluation.dataset import save_dataset
    from ideasim.evaluation.parlgov import (
        build_parlgov_dataset,
        download_and_build,
    )

    decade_aggregation = not args.per_year

    if args.election_csv and args.party_csv:
        # Use local files
        print(f"Using local CSVs: {args.election_csv}, {args.party_csv}")
        dataset = build_parlgov_dataset(
            election_csv=Path(args.election_csv),
            party_csv=Path(args.party_csv),
            decade_aggregation=decade_aggregation,
            dataset_name=args.name,
        )
    else:
        # Download from ParlGov
        cache_dir = (
            Path(args.cache_dir) if args.cache_dir else Path("data/parlgov_cache")
        )
        print(f"Downloading ParlGov data (cache: {cache_dir})...")
        dataset = download_and_build(
            cache_dir=cache_dir,
            decade_aggregation=decade_aggregation,
            dataset_name=args.name,
        )

    save_dataset(dataset, args.output)

    print(f"\nDataset: {dataset.name}")
    print(f"Years: {dataset.year_range[0]:.0f}-{dataset.year_range[1]:.0f}")
    print(f"Ideologies: {', '.join(dataset.ideology_names)}")
    print(f"Observations: {len(dataset.observations)}")
    agg_mode = "per-year" if args.per_year else "decade"
    print(f"Aggregation: {agg_mode}")

    # Print a summary table
    print("\n--- Shares by Period ---")
    for year in dataset.years:
        shares = dataset.shares_at_year(year)
        parts = [f"{k}: {v:.3f}" for k, v in sorted(shares.items())]
        print(f"  {int(year)}: {', '.join(parts)}")

    print(f"\nSaved to {args.output}")


def _run_sensitivity(args: argparse.Namespace) -> None:
    if args.config:
        print(f"Loading config from {args.config}...")
        config = load_config(args.config)
    else:
        print("Using default config...")
        config = make_default_config()

    # Validate config first
    validation = validate_config(config)
    if not validation.is_valid:
        print("Config validation FAILED:")
        print(validation.summary())
        sys.exit(1)

    run_oat_flag = not args.no_oat
    run_lhs_flag = not args.no_lhs
    deterministic = not args.stochastic

    if not run_oat_flag and not run_lhs_flag:
        print("ERROR: both --no-oat and --no-lhs specified; nothing to do.")
        sys.exit(1)

    analyses = []
    if run_oat_flag:
        analyses.append(f"OAT ({args.oat_steps} steps)")
    if run_lhs_flag:
        analyses.append(f"LHS ({args.lhs_samples} samples)")
    print(f"Running sensitivity analysis: {', '.join(analyses)}")
    print(f"  Deterministic: {deterministic}, Seed: {args.seed}")

    t0 = time.time()
    result = run_sensitivity(
        config,
        oat_steps=args.oat_steps,
        lhs_samples=args.lhs_samples,
        seed=args.seed,
        deterministic=deterministic,
        run_oat_analysis=run_oat_flag,
        run_lhs_analysis=run_lhs_flag,
    )
    elapsed = time.time() - t0

    if args.json:
        result_dict = result.to_dict()
        result_dict["elapsed_seconds"] = round(elapsed, 2)
        print(json.dumps(result_dict, indent=2))
    else:
        print(f"\nDone in {elapsed:.1f}s — {result.n_parameters} parameters analyzed.")

        if result.oat:
            print(f"\n--- OAT Results ({len(result.oat)} parameters) ---")
            for oat_param in result.oat:
                # Show range of final shares for the dominant ideology
                if oat_param.runs:
                    # Find ideology with highest variance across the sweep
                    ideo_names = list(oat_param.runs[0].final_shares.keys())
                    max_range = 0.0
                    max_range_name = ideo_names[0] if ideo_names else "unknown"
                    for name in ideo_names:
                        vals = [r.final_shares.get(name, 0.0) for r in oat_param.runs]
                        val_range = max(vals) - min(vals)
                        if val_range > max_range:
                            max_range = val_range
                            max_range_name = name
                    print(
                        f"  {oat_param.parameter_name}: "
                        f"most affected = {max_range_name} "
                        f"(range {max_range:.4f})"
                    )

        if result.lhs_stats:
            print(f"\n--- LHS Summary ({len(result.lhs_runs)} samples) ---")
            for stat in result.lhs_stats:
                print(
                    f"  {stat.metric_name}: "
                    f"mean={stat.mean:.4f}, std={stat.std:.4f}, "
                    f"range=[{stat.min:.4f}, {stat.max:.4f}]"
                )

    # Save full results to file
    with open(args.output, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nFull results saved to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    """Return the ``ideasim`` argument parser, with every subcommand."""
    parser = argparse.ArgumentParser(
        prog="ideasim",
        description="Ideological Dynamics Simulator — models competition between philosophical currents",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_simulate_parser(subparsers)
    _add_default_config_parser(subparsers)
    _add_evaluate_parser(subparsers)
    _add_validate_parser(subparsers)
    _add_check_parser(subparsers)
    _add_sensitivity_parser(subparsers)
    _add_ingest_parlgov_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "default-config":
        config = make_default_config()
        save_config(config, args.output)
        print(f"Default config written to {args.output}")
        return

    if args.command == "simulate":
        _run_simulate(args)

    if args.command == "evaluate":
        _run_evaluate(args)

    if args.command == "validate":
        _run_validate(args)

    if args.command == "check":
        _run_check(args)

    if args.command == "sensitivity":
        _run_sensitivity(args)

    if args.command == "ingest-parlgov":
        _run_ingest_parlgov(args)


if __name__ == "__main__":
    main()
