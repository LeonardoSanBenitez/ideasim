"""Input validation for simulation configs and datasets.

Validates structural consistency, numerical sanity, and constraint
satisfaction before a simulation runs. This prevents cryptic numpy
errors deep in the engine by catching problems early with clear
error messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ideasim.types import SimulationConfig


@dataclass
class ValidationError:
    """A single validation issue."""

    field: str
    message: str
    severity: str = "error"  # "error" or "warning"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.field}: {self.message}"


@dataclass
class ValidationResult:
    """Result of validating a SimulationConfig."""

    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no errors (warnings are acceptable)."""
        return len(self.errors) == 0

    def add_error(self, field_name: str, message: str) -> None:
        self.errors.append(ValidationError(field_name, message, "error"))

    def add_warning(self, field_name: str, message: str) -> None:
        self.warnings.append(ValidationError(field_name, message, "warning"))

    def summary(self) -> str:
        """Human-readable summary of validation results."""
        lines: list[str] = []
        if self.is_valid and not self.warnings:
            return "Validation passed: no issues found."
        for err in self.errors:
            lines.append(str(err))
        for warn in self.warnings:
            lines.append(str(warn))
        status = "FAILED" if not self.is_valid else "PASSED with warnings"
        lines.append(
            f"Validation {status}: "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)."
        )
        return "\n".join(lines)


def validate_config(config: SimulationConfig) -> ValidationResult:
    """Validate a SimulationConfig for structural and numerical correctness.

    Checks:
    - At least 1 ideology.
    - All array dimensions are consistent (N ideologies, M env vars).
    - Initial shares sum to 1 and are non-negative.
    - Initial latent values are non-negative.
    - Time range is valid (t_start < t_end).
    - dt is positive and not larger than total duration.
    - Matrices are the right shape (interaction, suppression, etc.).
    - Stochastic parameter matrices are the right shape.
    - Parameter values are in reasonable ranges (warnings for extreme values).

    Args:
        config: The SimulationConfig to validate.

    Returns:
        ValidationResult with any errors and warnings found.
    """
    result = ValidationResult()

    n = config.n_ideologies
    m = config.n_env_vars

    # --- Basic structure ---
    if n == 0:
        result.add_error("ideologies", "Must have at least 1 ideology.")
        return result  # Can't check further

    if m == 0:
        result.add_error(
            "environment.variable_names",
            "Must have at least 1 environment variable.",
        )

    # --- Ideology env_response dimensions ---
    for i, ideo in enumerate(config.ideologies):
        if len(ideo.env_response) != m:
            result.add_error(
                f"ideologies[{i}].env_response",
                f"Expected length {m}, got {len(ideo.env_response)}.",
            )

    # --- Initial shares ---
    if len(config.initial_shares) != n:
        result.add_error(
            "initial_shares",
            f"Expected length {n}, got {len(config.initial_shares)}.",
        )
    else:
        if np.any(config.initial_shares < 0):
            result.add_error(
                "initial_shares",
                "Contains negative values.",
            )
        share_sum = float(np.sum(config.initial_shares))
        if abs(share_sum - 1.0) > 1e-6:
            result.add_error(
                "initial_shares",
                f"Must sum to 1.0, got {share_sum:.8f}.",
            )

    # --- Initial latent ---
    if len(config.initial_latent) != n:
        result.add_error(
            "initial_latent",
            f"Expected length {n}, got {len(config.initial_latent)}.",
        )
    else:
        if np.any(config.initial_latent < 0):
            result.add_error(
                "initial_latent",
                "Contains negative values.",
            )

    # --- Time range ---
    if config.t_start >= config.t_end:
        result.add_error(
            "t_start/t_end",
            f"t_start ({config.t_start}) must be < t_end ({config.t_end}).",
        )

    if config.dt <= 0:
        result.add_error("dt", f"Must be positive, got {config.dt}.")
    elif config.dt > (config.t_end - config.t_start):
        result.add_error(
            "dt",
            f"dt ({config.dt}) is larger than total duration "
            f"({config.t_end - config.t_start}).",
        )

    if config.record_interval < 1:
        result.add_error(
            "record_interval",
            f"Must be >= 1, got {config.record_interval}.",
        )

    # --- Interaction matrix ---
    B = config.interaction_matrix
    if B.shape != (n, n):
        result.add_error(
            "interaction_matrix",
            f"Expected shape ({n}, {n}), got {B.shape}.",
        )

    # --- Suppression matrix ---
    S = config.suppression_matrix
    if S.shape != (n, n):
        result.add_error(
            "suppression_matrix",
            f"Expected shape ({n}, {n}), got {S.shape}.",
        )
    elif np.any(S < 0):
        result.add_warning(
            "suppression_matrix",
            "Contains negative values (unusual but not necessarily wrong).",
        )

    # --- Environment ---
    env = config.environment
    if len(env.initial) != m:
        result.add_error(
            "environment.initial",
            f"Expected length {m}, got {len(env.initial)}.",
        )

    if env.autonomous_decay is not None and len(env.autonomous_decay) != m:
        result.add_error(
            "environment.autonomous_decay",
            f"Expected length {m}, got {len(env.autonomous_decay)}.",
        )

    if env.policy_influence is not None:
        P = env.policy_influence
        if P.shape[0] < n:
            result.add_error(
                "environment.policy_influence",
                f"Expected at least {n} rows (one per ideology), got {P.shape[0]}.",
            )
        if P.shape[1] != m:
            result.add_error(
                "environment.policy_influence",
                f"Expected {m} columns (one per env var), got {P.shape[1]}.",
            )

    # Validate exogenous shocks
    for idx, (shock_t, var_idx, _delta) in enumerate(env.exogenous_shocks):
        if var_idx < 0 or var_idx >= m:
            result.add_error(
                f"environment.exogenous_shocks[{idx}]",
                f"variable_index {var_idx} out of range [0, {m}).",
            )
        if shock_t < config.t_start or shock_t > config.t_end:
            result.add_warning(
                f"environment.exogenous_shocks[{idx}]",
                f"Shock at t={shock_t} is outside simulation range "
                f"[{config.t_start}, {config.t_end}].",
            )

    # --- Stochastic parameters ---
    sp = config.stochastic
    if sp.hybridization_rates.shape != (n, n):
        result.add_error(
            "stochastic.hybridization_rates",
            f"Expected shape ({n}, {n}), got {sp.hybridization_rates.shape}.",
        )

    if sp.hybridization_thresholds.shape != (n, n):
        result.add_error(
            "stochastic.hybridization_thresholds",
            f"Expected shape ({n}, {n}), got {sp.hybridization_thresholds.shape}.",
        )

    # --- Warnings for extreme parameter values ---
    if config.rho < 0:
        result.add_error("rho", f"Must be non-negative, got {config.rho}.")
    elif config.rho > 0.5:
        result.add_warning(
            "rho",
            f"Value {config.rho} is unusually large (typical range: 0.001-0.1).",
        )

    for i, ideo in enumerate(config.ideologies):
        prefix = f"ideologies[{i}] ({ideo.name})"
        if ideo.latent_conversion < 0:
            result.add_error(f"{prefix}.latent_conversion", "Must be non-negative.")
        if ideo.latent_generation < 0:
            result.add_error(f"{prefix}.latent_generation", "Must be non-negative.")
        if ideo.latent_decay < 0:
            result.add_error(f"{prefix}.latent_decay", "Must be non-negative.")
        if ideo.capture_threshold < 0 or ideo.capture_threshold > 1:
            result.add_warning(
                f"{prefix}.capture_threshold",
                f"Value {ideo.capture_threshold} outside [0, 1] range.",
            )
        if ideo.governance_threshold < 0 or ideo.governance_threshold > 1:
            result.add_warning(
                f"{prefix}.governance_threshold",
                f"Value {ideo.governance_threshold} outside [0, 1] range.",
            )

    return result
