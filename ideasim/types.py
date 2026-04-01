"""Core data types for the Ideological Dynamics Simulator.

Based on Cidral's formal framework: replicator-mutator dynamics on the simplex,
latent sympathy for dormancy/martyrdom, endogenous environment modification,
hybridization as stochastic jump process. The full system is a PDMP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class Ideology:
    """A single ideological current in the ecosystem.

    Attributes:
        name: Human-readable name (e.g., "liberalism", "Marxism").
        env_response: Weight vector w_i (length M) — how this ideology responds
            to environmental conditions. Positive = thrives in that condition.
        latent_conversion: gamma_i — rate at which latent sympathy converts
            to manifest adoption.
        latent_generation: alpha_i — rate at which active adherents generate
            latent sympathizers.
        latent_decay: beta_i — natural decay rate of latent sympathy.
        martyrdom_coeff: delta_i — how efficiently suppression creates sympathy.
        institutional_amp: mu_i — amplification rate above capture threshold.
        capture_threshold: theta_i — share above which institutional capture kicks in.
        governance_threshold: theta_i^gov — share above which ideology can suppress others.
        mutation_rate: nu_i — rate of parameter drift per year.
    """

    name: str
    env_response: NDArray[np.float64]
    latent_conversion: float = 0.1
    latent_generation: float = 0.3
    latent_decay: float = 0.05
    martyrdom_coeff: float = 0.5
    institutional_amp: float = 0.2
    capture_threshold: float = 0.25
    governance_threshold: float = 0.4
    mutation_rate: float = 0.01


@dataclass
class StochasticParams:
    """Parameters for stochastic event overlays.

    Attributes:
        hybridization_rates: p_{ij} — rate of hybridization between i and j.
            Shape (N, N), symmetric. Zero on diagonal.
        hybridization_thresholds: H_{ij} — latent sympathy overlap threshold.
            Shape (N, N), symmetric.
        charisma_rate: lambda_char — global rate of charismatic events per year.
        charisma_decay: tau_char — decay time of charismatic effect (years).
        charisma_amplitude_mean: Mean of log-normal charisma amplitude.
        charisma_amplitude_std: Std of log-normal charisma amplitude.
        mutation_step_size: sigma_mut — std of Gaussian mutation perturbation.
        hybridization_seed_share: epsilon_h — initial share of new hybrid.
    """

    hybridization_rates: NDArray[np.float64]
    hybridization_thresholds: NDArray[np.float64]
    charisma_rate: float = 0.02
    charisma_decay: float = 20.0
    charisma_amplitude_mean: float = 0.3
    charisma_amplitude_std: float = 0.5
    mutation_step_size: float = 0.01
    hybridization_seed_share: float = 0.02


@dataclass
class EnvironmentSchedule:
    """Exogenous environment trajectory and endogenous feedback.

    Attributes:
        variable_names: Names of the M environmental variables.
        initial: e(0), shape (M,).
        exogenous_shocks: List of (time, variable_index, delta) — discrete shocks.
        autonomous_decay: g_k decay rate toward baseline for each variable.
        policy_influence: P matrix (N x M) — how ideologies change the environment.
    """

    variable_names: list[str]
    initial: NDArray[np.float64]
    exogenous_shocks: list[tuple[float, int, float]] = field(default_factory=list)
    autonomous_decay: NDArray[np.float64] | None = None
    policy_influence: NDArray[np.float64] | None = None


@dataclass
class SimulationConfig:
    """Complete configuration for an ideological dynamics simulation.

    Attributes:
        ideologies: List of ideological types.
        interaction_matrix: B — NxN competitive interaction matrix.
        suppression_matrix: s_{ij} — NxN, propensity of j to suppress i when in power.
        environment: Environment configuration.
        stochastic: Stochastic event parameters.
        initial_shares: x(0) on the simplex, shape (N,).
        initial_latent: y(0), shape (N,).
        rho: Replicator-mutator immigration rate (latent -> manifest).
        t_start: Start year.
        t_end: End year.
        dt: Integration time step (years).
        record_interval: How often to record state (in time steps).
        seed: Random seed for reproducibility.
    """

    ideologies: list[Ideology]
    interaction_matrix: NDArray[np.float64]
    suppression_matrix: NDArray[np.float64]
    environment: EnvironmentSchedule
    stochastic: StochasticParams
    initial_shares: NDArray[np.float64]
    initial_latent: NDArray[np.float64]
    rho: float = 0.005
    t_start: float = 1800.0
    t_end: float = 2025.0
    dt: float = 0.05
    record_interval: int = 20
    seed: int | None = 42

    @property
    def n_ideologies(self) -> int:
        return len(self.ideologies)

    @property
    def n_env_vars(self) -> int:
        return len(self.environment.variable_names)

    def ideology_names(self) -> list[str]:
        return [ideo.name for ideo in self.ideologies]


@dataclass
class StateSnapshot:
    """Immutable snapshot of the system at a single point in time."""

    time: float
    shares: NDArray[np.float64]
    latent: NDArray[np.float64]
    environment: NDArray[np.float64]
    ideology_names: list[str]
    fitness: NDArray[np.float64] | None = None


@dataclass
class SimulationEvent:
    """Record of a discrete stochastic event."""

    time: float
    event_type: str  # "hybridization", "mutation", "charisma"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Complete result of a simulation run."""

    config_summary: dict[str, Any]
    history: list[StateSnapshot]
    events: list[SimulationEvent]
    ideology_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.history)

    def times(self) -> NDArray[np.float64]:
        return np.array([s.time for s in self.history])

    def share_timeseries(self, name: str) -> NDArray[np.float64]:
        idx = self.ideology_names.index(name)
        return np.array([s.shares[idx] for s in self.history])

    def latent_timeseries(self, name: str) -> NDArray[np.float64]:
        idx = self.ideology_names.index(name)
        return np.array([s.latent[idx] for s in self.history])

    def env_timeseries(self, var_index: int) -> NDArray[np.float64]:
        return np.array([s.environment[var_index] for s in self.history])

    def entropy_timeseries(self) -> NDArray[np.float64]:
        """Shannon entropy of ideological distribution over time."""
        result = []
        for s in self.history:
            x = s.shares
            x_pos = x[x > 1e-15]
            result.append(-float(np.sum(x_pos * np.log(x_pos))))
        return np.array(result)
