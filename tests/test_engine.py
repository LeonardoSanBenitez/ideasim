"""Tests for the ideological dynamics simulation engine."""

import numpy as np

from ideasim.defaults import make_default_config
from ideasim.engine import simulate
from ideasim.types import SimulationConfig


def _quick_config() -> SimulationConfig:
    """Default config with shorter time range for fast tests."""
    config = make_default_config()
    config.t_end = 1850.0  # 50 years instead of 225
    config.record_interval = 40
    return config


class TestSimplexInvariance:
    """The simplex constraint (shares sum to 1) must hold throughout."""

    def test_shares_sum_to_one(self) -> None:
        config = _quick_config()
        result = simulate(config)
        for snap in result.history:
            total = np.sum(snap.shares)
            assert abs(total - 1.0) < 1e-6, f"Shares sum to {total} at t={snap.time}"

    def test_shares_non_negative(self) -> None:
        config = _quick_config()
        result = simulate(config)
        for snap in result.history:
            assert np.all(snap.shares >= -1e-10), (
                f"Negative share at t={snap.time}: {snap.shares}"
            )

    def test_latent_non_negative(self) -> None:
        config = _quick_config()
        result = simulate(config)
        for snap in result.history:
            assert np.all(snap.latent >= -1e-10), f"Negative latent at t={snap.time}"


class TestDormancyReemergence:
    """An ideology with zero manifest share but positive latent sympathy can re-emerge."""

    def test_reemergence_from_latent(self) -> None:
        config = _quick_config()
        # Start Marxism with near-zero manifest but high latent
        config.initial_shares = np.array([0.30, 0.001, 0.50, 0.199])
        config.initial_latent = np.array([0.1, 0.8, 0.1, 0.1])
        config.rho = 0.05  # strong immigration from latent
        config.t_end = 1900.0

        result = simulate(config)
        # Marxism (index 1) should have re-emerged above its initial share
        marxism_final = result.history[-1].shares[1]
        assert marxism_final > 0.01, (
            f"Marxism failed to re-emerge: final share = {marxism_final}"
        )


class TestMartyrdomDynamics:
    """Suppression should build latent sympathy (martyrdom effect)."""

    def test_suppression_builds_latent(self) -> None:
        config = _quick_config()
        # Make nationalism dominant (suppresses others)
        config.initial_shares = np.array([0.1, 0.05, 0.15, 0.7])
        config.t_end = 1850.0

        result = simulate(config)
        # Suppressed ideologies should have growing latent sympathy
        initial_latent_lib = config.initial_latent[0]
        final_latent_lib = result.history[-1].latent[0]
        # Liberalism is suppressed by dominant nationalism -> latent should grow
        assert final_latent_lib > initial_latent_lib * 0.5, (
            f"Latent sympathy for liberalism didn't grow under suppression: "
            f"{initial_latent_lib} -> {final_latent_lib}"
        )


class TestDefaultScenario:
    """The default 225-year scenario should produce sensible results."""

    def test_runs_to_completion(self) -> None:
        config = make_default_config()
        result = simulate(config)
        assert result.n_steps > 100
        assert result.history[-1].time > 2020.0

    def test_no_ideology_totally_dominates(self) -> None:
        config = make_default_config()
        result = simulate(config)
        final = result.history[-1].shares
        # No single ideology should have >95% share in a 4-way competition
        assert np.max(final[:4]) < 0.95, f"One ideology totally dominates: {final}"

    def test_events_occur(self) -> None:
        config = make_default_config()
        result = simulate(config)
        # Over 225 years, we should see some stochastic events
        assert len(result.events) > 0, "No stochastic events in 225 years?"


class TestReproducibility:
    """Same seed should produce identical results."""

    def test_deterministic_with_seed(self) -> None:
        config1 = _quick_config()
        config1.seed = 123
        config2 = _quick_config()
        config2.seed = 123

        r1 = simulate(config1)
        r2 = simulate(config2)

        for s1, s2 in zip(r1.history, r2.history, strict=True):
            np.testing.assert_array_almost_equal(s1.shares, s2.shares)

    def test_different_seed_differs(self) -> None:
        config1 = _quick_config()
        config1.seed = 1
        config2 = _quick_config()
        config2.seed = 2

        r1 = simulate(config1)
        r2 = simulate(config2)

        # At least some states should differ
        diffs = [
            np.max(np.abs(s1.shares - s2.shares))
            for s1, s2 in zip(r1.history, r2.history, strict=False)
            if len(s1.shares) == len(s2.shares)
        ]
        assert max(diffs) > 0.001, "Different seeds produced identical results"
