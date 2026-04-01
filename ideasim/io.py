"""JSON serialization/deserialization for simulation configs and results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ideasim.types import (
    EnvironmentSchedule,
    Ideology,
    SimulationConfig,
    SimulationResult,
    StochasticParams,
)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


def result_to_dict(result: SimulationResult) -> dict[str, Any]:
    """Convert SimulationResult to a JSON-serializable dict."""
    history_records = []
    for snap in result.history:
        record: dict[str, Any] = {
            "time": round(snap.time, 2),
            "shares": {
                name: round(float(snap.shares[i]), 6)
                for i, name in enumerate(snap.ideology_names)
                if i < len(snap.shares)
            },
            "latent": {
                name: round(float(snap.latent[i]), 6)
                for i, name in enumerate(snap.ideology_names)
                if i < len(snap.latent)
            },
            "environment": {
                name: round(float(snap.environment[i]), 4)
                for i, name in enumerate(result.config_summary.get("env_var_names", []))
                if i < len(snap.environment)
            },
        }
        if snap.fitness is not None:
            record["fitness"] = {
                name: round(float(snap.fitness[i]), 6)
                for i, name in enumerate(snap.ideology_names)
                if i < len(snap.fitness)
            }
        history_records.append(record)

    event_records = []
    for ev in result.events:
        event_records.append({
            "time": round(ev.time, 2),
            "type": ev.event_type,
            "details": ev.details,
        })

    return {
        "config": result.config_summary,
        "metadata": result.metadata,
        "ideology_names": result.ideology_names,
        "history": history_records,
        "events": event_records,
        "summary": _compute_summary(result),
    }


def _compute_summary(result: SimulationResult) -> dict[str, Any]:
    """Compute summary statistics for the simulation."""
    if not result.history:
        return {}

    first = result.history[0]
    last = result.history[-1]
    n_initial = len(first.ideology_names)

    summary: dict[str, Any] = {
        "duration_years": round(last.time - first.time, 1),
        "initial_ideologies": first.ideology_names,
        "final_ideologies": last.ideology_names,
        "hybridization_events": sum(1 for e in result.events if e.event_type == "hybridization"),
        "mutation_events": sum(1 for e in result.events if e.event_type == "mutation"),
        "charisma_events": sum(1 for e in result.events if e.event_type == "charisma"),
    }

    # Per-ideology summary
    ideology_summary = {}
    for i, name in enumerate(last.ideology_names):
        final_share = float(last.shares[i]) if i < len(last.shares) else 0.0
        initial_share = float(first.shares[i]) if i < n_initial and i < len(first.shares) else 0.0

        # Find peak share
        peak_share = 0.0
        peak_time = first.time
        for snap in result.history:
            if i < len(snap.shares) and float(snap.shares[i]) > peak_share:
                peak_share = float(snap.shares[i])
                peak_time = snap.time

        ideology_summary[name] = {
            "initial_share": round(initial_share, 4),
            "final_share": round(final_share, 4),
            "peak_share": round(peak_share, 4),
            "peak_year": round(peak_time, 1),
            "change": round(final_share - initial_share, 4),
        }

    summary["ideologies"] = ideology_summary

    # Entropy over time
    times = result.times()
    entropy = result.entropy_timeseries()
    if len(entropy) > 0:
        summary["entropy"] = {
            "initial": round(float(entropy[0]), 4),
            "final": round(float(entropy[-1]), 4),
            "max": round(float(np.max(entropy)), 4),
            "max_year": round(float(times[np.argmax(entropy)]), 1),
            "min": round(float(np.min(entropy)), 4),
            "min_year": round(float(times[np.argmin(entropy)]), 1),
        }

    return summary


def save_result(result: SimulationResult, path: str | Path) -> None:
    """Save simulation result to JSON file."""
    data = result_to_dict(result)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)


def load_config(path: str | Path) -> SimulationConfig:
    """Load simulation config from JSON file."""
    with open(path) as f:
        data = json.load(f)

    ideologies = []
    for ideo_data in data["ideologies"]:
        ideologies.append(
            Ideology(
                name=ideo_data["name"],
                env_response=np.array(ideo_data["env_response"]),
                latent_conversion=ideo_data.get("latent_conversion", 0.1),
                latent_generation=ideo_data.get("latent_generation", 0.3),
                latent_decay=ideo_data.get("latent_decay", 0.05),
                martyrdom_coeff=ideo_data.get("martyrdom_coeff", 0.5),
                institutional_amp=ideo_data.get("institutional_amp", 0.2),
                capture_threshold=ideo_data.get("capture_threshold", 0.25),
                governance_threshold=ideo_data.get("governance_threshold", 0.4),
                mutation_rate=ideo_data.get("mutation_rate", 0.01),
            )
        )

    n = len(ideologies)

    env_data = data["environment"]
    shocks = [
        (s["time"], s["variable_index"], s["delta"])
        for s in env_data.get("exogenous_shocks", [])
    ]

    env = EnvironmentSchedule(
        variable_names=env_data["variable_names"],
        initial=np.array(env_data["initial"]),
        exogenous_shocks=shocks,
        autonomous_decay=np.array(env_data["autonomous_decay"]) if "autonomous_decay" in env_data else None,
        policy_influence=np.array(env_data["policy_influence"]) if "policy_influence" in env_data else None,
    )

    stoch_data = data.get("stochastic", {})
    stochastic = StochasticParams(
        hybridization_rates=np.array(stoch_data.get("hybridization_rates", np.full((n, n), 0.005))),
        hybridization_thresholds=np.array(stoch_data.get("hybridization_thresholds", np.full((n, n), 0.1))),
        charisma_rate=stoch_data.get("charisma_rate", 0.02),
        charisma_decay=stoch_data.get("charisma_decay", 20.0),
        charisma_amplitude_mean=stoch_data.get("charisma_amplitude_mean", 0.3),
        charisma_amplitude_std=stoch_data.get("charisma_amplitude_std", 0.5),
        mutation_step_size=stoch_data.get("mutation_step_size", 0.01),
        hybridization_seed_share=stoch_data.get("hybridization_seed_share", 0.02),
    )

    return SimulationConfig(
        ideologies=ideologies,
        interaction_matrix=np.array(data["interaction_matrix"]),
        suppression_matrix=np.array(data["suppression_matrix"]),
        environment=env,
        stochastic=stochastic,
        initial_shares=np.array(data["initial_shares"]),
        initial_latent=np.array(data["initial_latent"]),
        rho=data.get("rho", 0.005),
        t_start=data.get("t_start", 1800.0),
        t_end=data.get("t_end", 2025.0),
        dt=data.get("dt", 0.05),
        record_interval=data.get("record_interval", 20),
        seed=data.get("seed", 42),
    )


def save_config(config: SimulationConfig, path: str | Path) -> None:
    """Save simulation config to JSON file."""
    data: dict[str, Any] = {
        "ideologies": [
            {
                "name": ideo.name,
                "env_response": ideo.env_response.tolist(),
                "latent_conversion": ideo.latent_conversion,
                "latent_generation": ideo.latent_generation,
                "latent_decay": ideo.latent_decay,
                "martyrdom_coeff": ideo.martyrdom_coeff,
                "institutional_amp": ideo.institutional_amp,
                "capture_threshold": ideo.capture_threshold,
                "governance_threshold": ideo.governance_threshold,
                "mutation_rate": ideo.mutation_rate,
            }
            for ideo in config.ideologies
        ],
        "interaction_matrix": config.interaction_matrix.tolist(),
        "suppression_matrix": config.suppression_matrix.tolist(),
        "environment": {
            "variable_names": config.environment.variable_names,
            "initial": config.environment.initial.tolist(),
            "exogenous_shocks": [
                {"time": t, "variable_index": vi, "delta": d}
                for t, vi, d in config.environment.exogenous_shocks
            ],
            "autonomous_decay": config.environment.autonomous_decay.tolist()
            if config.environment.autonomous_decay is not None
            else None,
            "policy_influence": config.environment.policy_influence.tolist()
            if config.environment.policy_influence is not None
            else None,
        },
        "stochastic": {
            "hybridization_rates": config.stochastic.hybridization_rates.tolist(),
            "hybridization_thresholds": config.stochastic.hybridization_thresholds.tolist(),
            "charisma_rate": config.stochastic.charisma_rate,
            "charisma_decay": config.stochastic.charisma_decay,
            "charisma_amplitude_mean": config.stochastic.charisma_amplitude_mean,
            "charisma_amplitude_std": config.stochastic.charisma_amplitude_std,
            "mutation_step_size": config.stochastic.mutation_step_size,
            "hybridization_seed_share": config.stochastic.hybridization_seed_share,
        },
        "initial_shares": config.initial_shares.tolist(),
        "initial_latent": config.initial_latent.tolist(),
        "rho": config.rho,
        "t_start": config.t_start,
        "t_end": config.t_end,
        "dt": config.dt,
        "record_interval": config.record_interval,
        "seed": config.seed,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
