"""Tests for the health check module."""

from __future__ import annotations

import numpy as np
import pytest

from ideasim.defaults import make_default_config
from ideasim.engine import simulate
from ideasim.health import HealthReport, check_health
from ideasim.types import SimulationResult, StateSnapshot


class TestHealthCheckDefaultScenario:
    """Health checks on the default scenario should all pass."""

    @pytest.fixture
    def default_result(self) -> SimulationResult:
        config = make_default_config()
        return simulate(config)

    def test_all_checks_pass(self, default_result: SimulationResult) -> None:
        report = check_health(default_result)
        assert report.all_passed, report.summary()

    def test_has_expected_checks(self, default_result: SimulationResult) -> None:
        report = check_health(default_result)
        check_names = {c.name for c in report.checks}
        expected = {
            "simplex_invariant",
            "shares_nonneg",
            "latent_nonneg",
            "no_nan",
            "no_inf",
            "no_collapse",
            "no_total_dominance",
            "entropy_nonzero",
            "sufficient_history",
            "valid_event_types",
        }
        assert expected.issubset(check_names)


class TestHealthCheckEdgeCases:
    def test_empty_history(self) -> None:
        result = SimulationResult(
            config_summary={},
            history=[],
            events=[],
            ideology_names=["a"],
        )
        report = check_health(result)
        assert not report.all_passed

    def test_nan_in_shares(self) -> None:
        snap = StateSnapshot(
            time=0.0,
            shares=np.array([np.nan, 0.5]),
            latent=np.array([0.1, 0.1]),
            environment=np.array([1.0]),
            ideology_names=["a", "b"],
        )
        result = SimulationResult(
            config_summary={},
            history=[snap] * 15,
            events=[],
            ideology_names=["a", "b"],
        )
        report = check_health(result)
        nan_check = next(c for c in report.checks if c.name == "no_nan")
        assert not nan_check.passed

    def test_total_dominance_warning(self) -> None:
        """A single ideology at ~1.0 should trigger a warning."""
        snap = StateSnapshot(
            time=0.0,
            shares=np.array([0.999, 0.001]),
            latent=np.array([0.1, 0.01]),
            environment=np.array([1.0]),
            ideology_names=["a", "b"],
        )
        # Need multiple snapshots for sufficient_history
        snaps = [snap] * 15
        result = SimulationResult(
            config_summary={},
            history=snaps,
            events=[],
            ideology_names=["a", "b"],
        )
        report = check_health(result)
        dominance_check = next(
            c for c in report.checks if c.name == "no_total_dominance"
        )
        assert not dominance_check.passed


class TestHealthReport:
    def test_to_dict_serializable(self) -> None:
        """Health report should be JSON-serializable."""
        config = make_default_config()
        result = simulate(config)
        report = check_health(result)
        d = report.to_dict()
        import json

        json.dumps(d)  # Should not raise

    def test_summary_all_passed(self) -> None:
        report = HealthReport()
        # No checks at all => all_passed is True
        assert "0 health checks" in report.summary()

    def test_n_errors_and_warnings(self) -> None:
        config = make_default_config()
        result = simulate(config)
        report = check_health(result)
        assert report.n_errors == 0
        assert report.n_warnings == 0
