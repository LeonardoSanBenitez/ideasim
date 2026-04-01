"""CLI entry point for the Ideological Dynamics Simulator."""

from __future__ import annotations

import argparse
import sys
import time

from ideasim.defaults import make_default_config
from ideasim.engine import simulate
from ideasim.io import load_config, save_config, save_result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ideasim",
        description="Ideological Dynamics Simulator — models competition between philosophical currents",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # simulate command
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

    # default-config command
    def_parser = subparsers.add_parser("default-config", help="Export the default config to JSON")
    def_parser.add_argument(
        "-o", "--output",
        default="default_config.json",
        help="Output path (default: default_config.json)",
    )

    args = parser.parse_args(argv)

    if args.command == "default-config":
        config = make_default_config()
        save_config(config, args.output)
        print(f"Default config written to {args.output}")
        return

    if args.command == "simulate":
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


if __name__ == "__main__":
    main()
