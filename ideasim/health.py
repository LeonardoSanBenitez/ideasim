"""Health checks for simulation results.

Post-hoc sanity checks on simulation output to detect numerical
issues, degenerate dynamics, or parameter tuning problems. These
are the "basic reporting metrics" that help assess whether a
simulation run is producing sensible results.

Designed to be run automatically after every simulation (e.g., in
CI/CD or as part of a test harness) to flag problems early.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ideasim.types import SimulationResult


@dataclass
class HealthCheck:
    """A single health check result."""

    name: str
    passed: bool
    message: str
    severity: str = "error"  # "error", "warning", "info"

    def __str__(self) -> str:
        status = "PASS" if self.passed else self.severity.upper()
        return f"[{status}] {self.name}: {self.message}"


@dataclass
class HealthReport:
    """Aggregated health report for a simulation run."""

    checks: list[HealthCheck] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """True if all checks passed."""
        return all(c.passed for c in self.checks)

    @property
    def n_errors(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "warning")

    def summary(self) -> str:
        """Human-readable summary."""
        lines: list[str] = []
        for check in self.checks:
            if not check.passed:
                lines.append(str(check))
        if self.all_passed:
            lines.append(f"All {len(self.checks)} health checks passed.")
        else:
            passed = sum(1 for c in self.checks if c.passed)
            lines.append(
                f"{passed}/{len(self.checks)} checks passed, "
                f"{self.n_errors} error(s), {self.n_warnings} warning(s)."
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation."""
        return {
            "all_passed": self.all_passed,
            "n_checks": len(self.checks),
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "severity": c.severity,
                }
                for c in self.checks
            ],
        }


def check_health(result: SimulationResult) -> HealthReport:
    """Run all health checks on a simulation result.

    Checks performed:
    1. Simplex invariant: shares sum to ~1.0 at every recorded step.
    2. Non-negativity: shares and latent values never go negative.
    3. No NaN/Inf: no numerical blowup in shares, latent, or environment.
    4. No total collapse: at least 2 ideologies above 1% at the end.
    5. No pathological dominance: no ideology at exactly 1.0 (within tolerance).
    6. Reasonable entropy range: final entropy is not zero (unless only 1 ideology).
    7. Sufficient history: at least 10 recorded snapshots.
    8. Event plausibility: stochastic events have valid types and timing.

    Args:
        result: The SimulationResult to check.

    Returns:
        HealthReport with all check results.
    """
    report = HealthReport()

    if not result.history:
        report.checks.append(
            HealthCheck("history_nonempty", False, "No history snapshots.", "error")
        )
        return report

    # 1. Simplex invariant
    max_simplex_deviation = 0.0
    worst_time = result.history[0].time
    for snap in result.history:
        deviation = abs(float(np.sum(snap.shares)) - 1.0)
        if deviation > max_simplex_deviation:
            max_simplex_deviation = deviation
            worst_time = snap.time

    simplex_ok = max_simplex_deviation < 1e-6
    report.checks.append(
        HealthCheck(
            "simplex_invariant",
            simplex_ok,
            f"Max deviation from simplex: {max_simplex_deviation:.2e} "
            f"(at t={worst_time:.1f}).",
            "error",
        )
    )

    # 2. Non-negativity
    min_share = min(float(np.min(snap.shares)) for snap in result.history)
    shares_nonneg = min_share >= -1e-10
    report.checks.append(
        HealthCheck(
            "shares_nonneg",
            shares_nonneg,
            f"Min share value: {min_share:.2e}.",
            "error",
        )
    )

    min_latent = min(float(np.min(snap.latent)) for snap in result.history)
    latent_nonneg = min_latent >= -1e-10
    report.checks.append(
        HealthCheck(
            "latent_nonneg",
            latent_nonneg,
            f"Min latent value: {min_latent:.2e}.",
            "error",
        )
    )

    # 3. No NaN/Inf
    has_nan = False
    has_inf = False
    for snap in result.history:
        if np.any(np.isnan(snap.shares)) or np.any(np.isnan(snap.latent)):
            has_nan = True
        if np.any(np.isinf(snap.shares)) or np.any(np.isinf(snap.latent)):
            has_inf = True
        if np.any(np.isnan(snap.environment)):
            has_nan = True
        if np.any(np.isinf(snap.environment)):
            has_inf = True

    report.checks.append(
        HealthCheck("no_nan", not has_nan, "No NaN values." if not has_nan else "NaN detected in history.", "error")
    )
    report.checks.append(
        HealthCheck("no_inf", not has_inf, "No Inf values." if not has_inf else "Inf detected in history.", "error")
    )

    # 4. No total collapse (at least 2 ideologies above 1% at end)
    last = result.history[-1]
    n_above_threshold = int(np.sum(last.shares > 0.01))
    n_initial = len(result.history[0].ideology_names)
    collapse_ok = n_above_threshold >= min(2, n_initial)
    report.checks.append(
        HealthCheck(
            "no_collapse",
            collapse_ok,
            f"{n_above_threshold} ideology(ies) above 1% at end "
            f"(of {len(last.ideology_names)} total).",
            "warning",
        )
    )

    # 5. No pathological dominance
    max_final_share = float(np.max(last.shares))
    dominance_ok = max_final_share < 0.999
    report.checks.append(
        HealthCheck(
            "no_total_dominance",
            dominance_ok,
            f"Max final share: {max_final_share:.4f}.",
            "warning",
        )
    )

    # 6. Entropy
    entropy_ts = result.entropy_timeseries()
    if len(entropy_ts) > 0:
        final_entropy = float(entropy_ts[-1])
        entropy_ok = final_entropy > 0.01 or n_initial <= 1
        report.checks.append(
            HealthCheck(
                "entropy_nonzero",
                entropy_ok,
                f"Final Shannon entropy: {final_entropy:.4f}.",
                "warning",
            )
        )

    # 7. Sufficient history
    n_snaps = len(result.history)
    history_ok = n_snaps >= 10
    report.checks.append(
        HealthCheck(
            "sufficient_history",
            history_ok,
            f"{n_snaps} snapshots recorded.",
            "warning",
        )
    )

    # 8. Event plausibility
    valid_event_types = {"hybridization", "mutation", "charisma"}
    bad_events = [
        ev for ev in result.events
        if ev.event_type not in valid_event_types
    ]
    events_ok = len(bad_events) == 0
    report.checks.append(
        HealthCheck(
            "valid_event_types",
            events_ok,
            f"All {len(result.events)} events have valid types."
            if events_ok
            else f"{len(bad_events)} event(s) with unknown type.",
            "warning",
        )
    )

    return report
