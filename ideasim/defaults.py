"""Default scenario: 4 ideologies in Europe, 1800-2025.

Parameters from Cidral's formalization (Section 7.3-7.6),
validated against Jose's mechanisms and Mark's historical cases.
"""

from __future__ import annotations

import numpy as np

from ideasim.types import (
    EnvironmentSchedule,
    Ideology,
    SimulationConfig,
    StochasticParams,
)

# Environmental variables: GDP growth, inequality, media freedom, social media, conflict
ENV_NAMES = ["gdp_growth", "inequality", "media_freedom", "social_media", "conflict"]

# Environmental response vectors (w_i): how each ideology responds to conditions
# Rows: liberalism, Marxism, traditionalism, nationalism
# Cols: GDP growth, inequality, media freedom, social media, conflict
# Scaled down: environmental influence should be gentle, not dominating
W = np.array(
    [
        [0.04, -0.01, 0.04, 0.01, -0.03],  # liberalism: thrives in growth + media freedom
        [-0.03, 0.05, -0.01, 0.02, 0.01],  # Marxism: thrives in inequality
        [0.01, -0.01, -0.02, -0.04, 0.03],  # traditionalism: harmed by social media, benefits from conflict
        [-0.01, 0.04, -0.03, 0.04, 0.04],  # nationalism: thrives in inequality, social media, conflict
    ]
)

# Competitive interaction matrix B
# Weakened: direct ideological competition is subtle, not a steamroller
B = np.array(
    [
        [0.0, -0.02, -0.01, -0.03],  # liberalism: harmed most by nationalism
        [-0.02, 0.0, -0.02, 0.01],  # Marxism: fed slightly by nationalism
        [-0.01, -0.03, 0.0, 0.02],  # traditionalism: harmed by Marxism, helped by nationalism
        [-0.02, 0.02, 0.02, 0.0],  # nationalism: feeds on Marxism and traditionalism
    ]
)

# Suppression matrix: propensity of j (col) to suppress i (row) when in power
S_MATRIX = np.array(
    [
        [0.0, 0.08, 0.03, 0.06],  # liberalism suppressed by Marxism, nationalism
        [0.03, 0.0, 0.06, 0.08],  # Marxism suppressed by traditionalism, nationalism
        [0.03, 0.08, 0.0, 0.03],  # traditionalism suppressed by Marxism
        [0.06, 0.06, 0.03, 0.0],  # nationalism suppressed by liberalism, Marxism
    ]
)

# Policy influence matrix P (N x M): how ideologies change the environment
# Policy influence: weak endogenous feedback (ideas change the world slowly)
P = np.array(
    [
        [0.01, -0.005, 0.01, 0.005, -0.005],  # liberalism: boosts growth, media freedom
        [-0.005, 0.003, -0.008, 0.0, 0.003],  # Marxism: reduces media freedom slightly
        [-0.003, 0.0, -0.005, -0.003, 0.003],  # traditionalism: slightly reduces freedoms
        [-0.005, 0.005, -0.01, 0.003, 0.008],  # nationalism: reduces media freedom, increases conflict
    ]
)

# Major exogenous shocks (time, env_variable_index, delta)
HISTORICAL_SHOCKS: list[tuple[float, int, float]] = [
    # French Revolution / Napoleonic Wars
    (1789.0, 4, 0.4),  # conflict spike
    (1789.0, 2, 0.2),  # media freedom burst
    # 1848 Revolutions
    (1848.0, 4, 0.3),
    (1848.0, 1, 0.2),  # inequality perception spike
    # Industrial Revolution peak
    (1870.0, 0, 0.15),  # GDP growth
    (1870.0, 1, 0.3),  # inequality surge
    # WWI
    (1914.0, 4, 0.6),
    (1914.0, 0, -0.3),
    # Russian Revolution
    (1917.0, 1, 0.3),
    # Great Depression
    (1929.0, 0, -0.5),
    (1929.0, 1, 0.4),
    # WWII
    (1939.0, 4, 0.7),
    (1939.0, 0, -0.2),
    # Post-war boom
    (1945.0, 4, -0.5),
    (1950.0, 0, 0.4),
    (1950.0, 2, 0.3),
    # 1960s liberalization
    (1965.0, 2, 0.2),
    # Oil crisis / stagflation
    (1973.0, 0, -0.25),
    # Fall of Berlin Wall
    (1989.0, 2, 0.3),
    (1989.0, 0, 0.2),
    # Internet era
    (1995.0, 3, 0.2),  # social media begins
    # 2008 financial crisis
    (2008.0, 0, -0.4),
    (2008.0, 1, 0.3),
    # Social media explosion
    (2012.0, 3, 0.5),
    # COVID + inflation
    (2020.0, 0, -0.3),
    (2020.0, 4, 0.2),
]


def make_default_ideologies() -> list[Ideology]:
    """Create the default 4-ideology system."""
    return [
        Ideology(
            name="liberalism",
            env_response=W[0],
            latent_conversion=0.05,
            latent_generation=0.15,
            latent_decay=0.05,
            martyrdom_coeff=0.3,
            institutional_amp=0.08,
            capture_threshold=0.30,
            governance_threshold=0.45,
            mutation_rate=0.01,
        ),
        Ideology(
            name="marxism",
            env_response=W[1],
            latent_conversion=0.06,
            latent_generation=0.18,
            latent_decay=0.04,
            martyrdom_coeff=0.5,
            institutional_amp=0.10,
            capture_threshold=0.25,
            governance_threshold=0.40,
            mutation_rate=0.015,
        ),
        Ideology(
            name="traditionalism",
            env_response=W[2],
            latent_conversion=0.04,
            latent_generation=0.12,
            latent_decay=0.03,
            martyrdom_coeff=0.3,
            institutional_amp=0.06,
            capture_threshold=0.35,
            governance_threshold=0.50,
            mutation_rate=0.005,
        ),
        Ideology(
            name="nationalism",
            env_response=W[3],
            latent_conversion=0.07,
            latent_generation=0.15,
            latent_decay=0.06,
            martyrdom_coeff=0.4,
            institutional_amp=0.12,
            capture_threshold=0.25,
            governance_threshold=0.40,
            mutation_rate=0.02,
        ),
    ]


def make_default_config() -> SimulationConfig:
    """Create the default European 1800-2025 simulation config."""
    n = 4
    ideologies = make_default_ideologies()

    hyb_rates = np.full((n, n), 0.0002)  # very rare: ~1 per 250 years per pair
    np.fill_diagonal(hyb_rates, 0.0)

    hyb_thresholds = np.full((n, n), 1.0)  # high threshold: needs substantial overlap
    np.fill_diagonal(hyb_thresholds, np.inf)

    stochastic = StochasticParams(
        hybridization_rates=hyb_rates,
        hybridization_thresholds=hyb_thresholds,
        charisma_rate=0.02,
        charisma_decay=20.0,
        charisma_amplitude_mean=0.3,
        charisma_amplitude_std=0.5,
        mutation_step_size=0.01,
        hybridization_seed_share=0.02,
    )

    env = EnvironmentSchedule(
        variable_names=ENV_NAMES,
        initial=np.array([0.02, 0.6, 0.3, 0.0, 0.2]),
        exogenous_shocks=HISTORICAL_SHOCKS,
        autonomous_decay=np.array([0.02, 0.01, 0.01, 0.0, 0.05]),
        policy_influence=P,
    )

    return SimulationConfig(
        ideologies=ideologies,
        interaction_matrix=B,
        suppression_matrix=S_MATRIX,
        environment=env,
        stochastic=stochastic,
        initial_shares=np.array([0.15, 0.05, 0.60, 0.20]),
        initial_latent=np.array([0.4, 0.3, 0.5, 0.4]),  # all have deep latent roots
        rho=0.015,  # strong latent-manifest coupling prevents monopoly
        t_start=1800.0,
        t_end=2025.0,
        dt=0.05,
        record_interval=20,  # record every 1 year (20 * 0.05)
        seed=42,
    )
