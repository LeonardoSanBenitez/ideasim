"""PDMP simulation engine for ideological dynamics.

Implements Cidral's framework: continuous replicator-mutator ODE dynamics
punctuated by discrete stochastic events (hybridization, mutation, charisma).
Integration via RK4; events via Poisson process overlay.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ideasim.types import (
    Ideology,
    SimulationConfig,
    SimulationEvent,
    SimulationResult,
    StateSnapshot,
)


def _precompute_ideology_params(
    ideologies: list[Ideology],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Extract ideology parameters into arrays for vectorized computation."""
    n = len(ideologies)
    m = len(ideologies[0].env_response)
    W = np.zeros((n, m))
    gamma = np.zeros(n)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    delta = np.zeros(n)
    mu = np.zeros(n)
    theta = np.zeros(n)
    theta_gov = np.zeros(n)
    sigma_scale = np.zeros(n)  # for crisis recruitment
    for i, ideo in enumerate(ideologies):
        W[i] = ideo.env_response
        gamma[i] = ideo.latent_conversion
        alpha[i] = ideo.latent_generation
        beta[i] = ideo.latent_decay
        delta[i] = ideo.martyrdom_coeff
        mu[i] = ideo.institutional_amp
        theta[i] = ideo.capture_threshold
        theta_gov[i] = ideo.governance_threshold
    return W, gamma, alpha, beta, delta, mu, theta, theta_gov, sigma_scale


def _compute_fitness(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    e: NDArray[np.float64],
    W: NDArray[np.float64],
    gamma: NDArray[np.float64],
    mu: NDArray[np.float64],
    theta: NDArray[np.float64],
    theta_gov: NDArray[np.float64],
    B: NDArray[np.float64],
    S_matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute fitness f_i for each ideology and suppression S_i (vectorized)."""
    n = len(x)
    # Suppression: S_i = sum_j s_{ij} * max(0, x_j - theta_j^gov)
    excess = np.maximum(x[:n] - theta_gov[:n], 0.0)
    suppression = S_matrix[:n, :n] @ excess

    # Fitness terms (all vectorized)
    f_base = W[:n] @ e                      # environmental
    f_compete = B[:n, :n] @ x[:n]           # competitive interaction
    f_latent = gamma[:n] * y[:n]            # latent reservoir
    f_inst = mu[:n] * np.maximum(x[:n] - theta[:n], 0.0)  # institutional

    f = f_base + f_compete + f_latent + f_inst
    return f, suppression


def _compute_derivatives(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    e: NDArray[np.float64],
    params: tuple[NDArray[np.float64], ...],
    config: SimulationConfig,
    active_charisma: list[tuple[float, int, float, float]],
    t: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute dx/dt, dy/dt, de/dt for the coupled ODE system (vectorized)."""
    W, gamma, alpha, beta, delta, mu, theta, theta_gov, _ = params
    n = len(x)
    m = len(e)

    f, suppression = _compute_fitness(
        x, y, e, W, gamma, mu, theta, theta_gov,
        config.interaction_matrix, config.suppression_matrix,
    )

    # Add active charismatic boosts
    for t_event, ideo_idx, amplitude, decay in active_charisma:
        if ideo_idx < n:
            f[ideo_idx] += amplitude * np.exp(-(t - t_event) / decay)

    f_bar = float(np.dot(x[:n], f))
    Y = float(np.sum(y[:n]))

    # dx/dt: replicator-mutator with latent immigration (Cidral eq 8.2)
    dx = x[:n] * (f - f_bar) + config.rho * (y[:n] - x[:n] * Y)

    # dy/dt: latent sympathy dynamics (vectorized)
    sigma = np.maximum(W[:n] @ e, 0.0) * 0.1
    dy = alpha[:n] * x[:n] - beta[:n] * y[:n] + sigma * (1.0 - x[:n]) + delta[:n] * suppression

    # de/dt: environment dynamics (Cidral eq 5.1)
    de = np.zeros(m)
    env = config.environment
    if env.autonomous_decay is not None:
        de -= env.autonomous_decay * (e - env.initial)
    if env.policy_influence is not None:
        P = env.policy_influence
        if P.shape[0] >= n:
            de += x[:n] @ P[:n]

    return dx, dy, de


def _rk4_step(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    e: NDArray[np.float64],
    params: tuple[NDArray[np.float64], ...],
    config: SimulationConfig,
    active_charisma: list[tuple[float, int, float, float]],
    t: float,
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Single RK4 integration step."""

    def derivs(
        x_: NDArray[np.float64],
        y_: NDArray[np.float64],
        e_: NDArray[np.float64],
        t_: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return _compute_derivatives(x_, y_, e_, params, config, active_charisma, t_)

    k1x, k1y, k1e = derivs(x, y, e, t)
    k2x, k2y, k2e = derivs(
        x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, e + 0.5 * dt * k1e, t + 0.5 * dt
    )
    k3x, k3y, k3e = derivs(
        x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, e + 0.5 * dt * k2e, t + 0.5 * dt
    )
    k4x, k4y, k4e = derivs(
        x + dt * k3x, y + dt * k3y, e + dt * k3e, t + dt
    )

    x_new = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    y_new = y + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
    e_new = e + (dt / 6.0) * (k1e + 2 * k2e + 2 * k3e + k4e)

    # Project x back to simplex (numerical safety)
    x_new = np.maximum(x_new, 0.0)
    x_new /= np.sum(x_new)

    # Ensure latent sympathy non-negative
    y_new = np.maximum(y_new, 0.0)

    return x_new, y_new, e_new


def _apply_exogenous_shocks(
    e: NDArray[np.float64],
    t: float,
    dt: float,
    shocks: list[tuple[float, int, float]],
) -> NDArray[np.float64]:
    """Apply any shocks that fall within [t, t+dt)."""
    e_new = e.copy()
    for shock_t, var_idx, delta in shocks:
        if t <= shock_t < t + dt and var_idx < len(e_new):
            e_new[var_idx] += delta
    return e_new


def _try_hybridization(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    ideologies: list[Ideology],
    config: SimulationConfig,
    rng: np.random.Generator,
    t: float,
    dt: float,
) -> SimulationEvent | None:
    """Check for and execute hybridization events."""
    n = len(ideologies)
    sp = config.stochastic

    for i in range(n):
        for j in range(i + 1, n):
            if i >= sp.hybridization_rates.shape[0] or j >= sp.hybridization_rates.shape[1]:
                continue
            overlap = y[i] * y[j]
            if overlap <= sp.hybridization_thresholds[i, j]:
                continue
            rate = sp.hybridization_rates[i, j]
            if rng.random() >= rate * dt:
                continue

            # Hybridization happens!
            lam = rng.uniform(0.3, 0.7)
            parent_a = ideologies[i]
            parent_b = ideologies[j]

            new_env = lam * parent_a.env_response + (1 - lam) * parent_b.env_response
            new_name = f"{parent_a.name[:4]}_{parent_b.name[:4]}"

            new_ideo = Ideology(
                name=new_name,
                env_response=new_env,
                latent_conversion=(parent_a.latent_conversion + parent_b.latent_conversion) / 2,
                latent_generation=(parent_a.latent_generation + parent_b.latent_generation) / 2,
                latent_decay=(parent_a.latent_decay + parent_b.latent_decay) / 2,
                martyrdom_coeff=(parent_a.martyrdom_coeff + parent_b.martyrdom_coeff) / 2,
                institutional_amp=max(parent_a.institutional_amp, parent_b.institutional_amp),
                capture_threshold=max(parent_a.capture_threshold, parent_b.capture_threshold),
                governance_threshold=max(parent_a.governance_threshold, parent_b.governance_threshold),
                mutation_rate=(parent_a.mutation_rate + parent_b.mutation_rate) / 2,
            )

            eps = sp.hybridization_seed_share
            # Extend state vectors
            ideologies.append(new_ideo)
            new_n = len(ideologies)

            # Extend interaction matrix
            old_B = config.interaction_matrix
            new_B = np.zeros((new_n, new_n))
            new_B[: old_B.shape[0], : old_B.shape[1]] = old_B
            new_B[-1, :new_n - 1] = lam * old_B[i, :] + (1 - lam) * old_B[j, :]  # type: ignore[index]
            new_B[:new_n - 1, -1] = lam * old_B[:, i] + (1 - lam) * old_B[:, j]  # type: ignore[index]
            config.interaction_matrix = new_B

            # Extend suppression matrix
            old_S = config.suppression_matrix
            new_S = np.zeros((new_n, new_n))
            new_S[: old_S.shape[0], : old_S.shape[1]] = old_S
            new_S[-1, :new_n - 1] = (lam * old_S[i, :] + (1 - lam) * old_S[j, :])  # type: ignore[index]
            new_S[:new_n - 1, -1] = (lam * old_S[:, i] + (1 - lam) * old_S[:, j])  # type: ignore[index]
            config.suppression_matrix = new_S

            # Extend stochastic matrices
            old_hr = sp.hybridization_rates
            new_hr = np.zeros((new_n, new_n))
            new_hr[: old_hr.shape[0], : old_hr.shape[1]] = old_hr
            new_hr[-1, :new_n - 1] = 0.005
            new_hr[:new_n - 1, -1] = 0.005
            sp.hybridization_rates = new_hr

            old_ht = sp.hybridization_thresholds
            new_ht = np.full((new_n, new_n), 0.1)
            new_ht[: old_ht.shape[0], : old_ht.shape[1]] = old_ht
            np.fill_diagonal(new_ht, np.inf)
            sp.hybridization_thresholds = new_ht

            # Extend policy influence
            if config.environment.policy_influence is not None:
                old_P = config.environment.policy_influence
                new_P = np.zeros((new_n, old_P.shape[1]))
                new_P[: old_P.shape[0], :] = old_P
                new_P[-1, :] = lam * old_P[i, :] + (1 - lam) * old_P[j, :]
                config.environment.policy_influence = new_P

            # Update shares: draw from parents
            x_new = np.zeros(new_n)
            x_new[:len(x)] = x
            half_eps = eps / 2
            x_new[i] -= half_eps
            x_new[j] -= half_eps
            x_new[-1] = eps
            x_new = np.maximum(x_new, 0.0)
            x_new /= np.sum(x_new)

            y_new = np.zeros(new_n)
            y_new[:len(y)] = y
            y_new[-1] = (y[i] + y[j]) * 0.1

            return SimulationEvent(
                time=t,
                event_type="hybridization",
                details={
                    "parents": [parent_a.name, parent_b.name],
                    "child": new_name,
                    "lambda": float(lam),
                    "x": x_new.tolist(),
                    "y": y_new.tolist(),
                },
            )

    return None


def _try_mutation(
    ideologies: list[Ideology],
    config: SimulationConfig,
    rng: np.random.Generator,
    t: float,
    dt: float,
) -> SimulationEvent | None:
    """Check for and execute mutation events."""
    for i, ideo in enumerate(ideologies):
        if rng.random() < ideo.mutation_rate * dt:
            perturbation = rng.normal(0, config.stochastic.mutation_step_size, size=ideo.env_response.shape)
            ideo.env_response = ideo.env_response + perturbation
            return SimulationEvent(
                time=t,
                event_type="mutation",
                details={
                    "ideology": ideo.name,
                    "perturbation": perturbation.tolist(),
                },
            )
    return None


def _try_charisma(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    ideologies: list[Ideology],
    config: SimulationConfig,
    rng: np.random.Generator,
    t: float,
    dt: float,
) -> SimulationEvent | None:
    """Check for and execute charismatic events."""
    sp = config.stochastic
    if rng.random() >= sp.charisma_rate * dt:
        return None

    # Choose ideology proportional to x_i * y_i
    n = len(ideologies)
    weights = x[:n] * y[:n]
    total = np.sum(weights)
    if total < 1e-15:
        return None
    probs = weights / total
    chosen = int(rng.choice(n, p=probs))

    amplitude = rng.lognormal(
        np.log(sp.charisma_amplitude_mean), sp.charisma_amplitude_std
    )

    return SimulationEvent(
        time=t,
        event_type="charisma",
        details={
            "ideology": ideologies[chosen].name,
            "amplitude": float(amplitude),
            "decay": sp.charisma_decay,
            "ideology_index": chosen,
        },
    )


def simulate(config: SimulationConfig) -> SimulationResult:
    """Run the full PDMP simulation.

    Returns a SimulationResult with time series and event log.
    """
    rng = np.random.default_rng(config.seed)

    x = config.initial_shares.copy()
    y = config.initial_latent.copy()
    e = config.environment.initial.copy()
    ideologies = list(config.ideologies)

    history: list[StateSnapshot] = []
    events: list[SimulationEvent] = []
    active_charisma: list[tuple[float, int, float, float]] = []

    t = config.t_start
    step = 0

    # Precompute ideology parameters for vectorized ops
    params = _precompute_ideology_params(ideologies)
    W, gamma, alpha, beta, delta, mu, theta, theta_gov, _ = params

    # Record initial state
    f_init, _ = _compute_fitness(
        x, y, e, W, gamma, mu, theta, theta_gov,
        config.interaction_matrix, config.suppression_matrix,
    )
    history.append(
        StateSnapshot(
            time=t,
            shares=x.copy(),
            latent=y.copy(),
            environment=e.copy(),
            ideology_names=[ideo.name for ideo in ideologies],
            fitness=f_init,
        )
    )

    while t < config.t_end:
        dt = config.dt

        # Apply exogenous shocks
        e = _apply_exogenous_shocks(e, t, dt, config.environment.exogenous_shocks)

        # Clean up expired charismatic effects (> 5 decay constants)
        active_charisma = [
            (te, idx, amp, dec)
            for te, idx, amp, dec in active_charisma
            if (t - te) < 5 * dec
        ]

        # RK4 integration step
        x, y, e = _rk4_step(x, y, e, params, config, active_charisma, t, dt)

        # Stochastic events
        hyb_event = _try_hybridization(x, y, ideologies, config, rng, t, dt)
        if hyb_event is not None:
            events.append(hyb_event)
            x = np.array(hyb_event.details["x"])
            y = np.array(hyb_event.details["y"])
            # Recompute params after adding new ideology
            params = _precompute_ideology_params(ideologies)
            W, gamma, alpha, beta, delta, mu, theta, theta_gov, _ = params

        mut_event = _try_mutation(ideologies, config, rng, t, dt)
        if mut_event is not None:
            events.append(mut_event)

        char_event = _try_charisma(x, y, ideologies, config, rng, t, dt)
        if char_event is not None:
            events.append(char_event)
            active_charisma.append((
                t,
                char_event.details["ideology_index"],
                char_event.details["amplitude"],
                char_event.details["decay"],
            ))

        t += dt
        step += 1

        # Record state at intervals
        if step % config.record_interval == 0:
            f_now, _ = _compute_fitness(
                x, y, e, W, gamma, mu, theta, theta_gov,
                config.interaction_matrix, config.suppression_matrix,
            )
            history.append(
                StateSnapshot(
                    time=t,
                    shares=x.copy(),
                    latent=y.copy(),
                    environment=e.copy(),
                    ideology_names=[ideo.name for ideo in ideologies],
                    fitness=f_now,
                )
            )

    # Config summary for JSON serialization
    config_summary = {
        "ideologies": [ideo.name for ideo in config.ideologies],
        "t_start": config.t_start,
        "t_end": config.t_end,
        "dt": config.dt,
        "rho": config.rho,
        "seed": config.seed,
        "n_env_vars": config.n_env_vars,
        "env_var_names": config.environment.variable_names,
    }

    return SimulationResult(
        config_summary=config_summary,
        history=history,
        events=events,
        ideology_names=[ideo.name for ideo in ideologies],
        metadata={
            "total_steps": step,
            "total_events": len(events),
            "final_n_ideologies": len(ideologies),
        },
    )
