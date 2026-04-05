"""Tests for the sensitivity analysis module."""

from __future__ import annotations

import json

import pytest

from ideasim.defaults import make_default_config
from ideasim.engine import simulate
from ideasim.evaluation.fitting import build_parameter_spec
from ideasim.sensitivity import (
    _compute_lhs_stats,
    _summarize_run,
    run_lhs,
    run_oat,
    run_sensitivity,
)
from ideasim.types import SimulationConfig


def _fast_config() -> SimulationConfig:
    """Short simulation for fast tests: 10 years, large dt."""
    config = make_default_config()
    config.t_start = 1800.0
    config.t_end = 1810.0
    config.dt = 0.5
    config.record_interval = 2
    config.seed = 42
    # Disable stochastic for speed and determinism
    config.stochastic.charisma_rate = 0.0
    config.stochastic.hybridization_rates[:] = 0.0
    for ideo in config.ideologies:
        ideo.mutation_rate = 0.0
    return config


class TestRunSummary:
    """RunSummary correctly distills a SimulationResult."""

    def test_summarize_run_has_final_shares(self) -> None:
        config = _fast_config()
        result = simulate(config)
        summary = _summarize_run(result, {"test_param": 1.0})
        assert len(summary.final_shares) == len(config.ideologies)
        # Shares should sum to ~1
        total = sum(summary.final_shares.values())
        assert abs(total - 1.0) < 1e-4

    def test_summarize_run_has_entropy(self) -> None:
        config = _fast_config()
        result = simulate(config)
        summary = _summarize_run(result, {})
        # 4 ideologies => entropy should be positive
        assert summary.final_entropy > 0.0

    def test_summarize_run_has_health(self) -> None:
        config = _fast_config()
        result = simulate(config)
        summary = _summarize_run(result, {})
        assert summary.health is not None
        assert isinstance(summary.health.all_passed, bool)

    def test_to_dict_is_json_serializable(self) -> None:
        config = _fast_config()
        result = simulate(config)
        summary = _summarize_run(result, {"rho": 0.005})
        d = summary.to_dict()
        # Should not raise
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


class TestOAT:
    """One-At-a-Time sensitivity analysis."""

    def test_oat_returns_one_result_per_parameter(self) -> None:
        config = _fast_config()
        params_spec = build_parameter_spec(config)
        results = run_oat(config, n_steps=3)
        assert len(results) == len(params_spec)

    def test_oat_each_result_has_correct_n_runs(self) -> None:
        config = _fast_config()
        results = run_oat(config, n_steps=4)
        for oat_result in results:
            assert len(oat_result.runs) == 4
            assert len(oat_result.values) == 4

    def test_oat_values_span_bounds(self) -> None:
        config = _fast_config()
        params_spec = build_parameter_spec(config)
        results = run_oat(config, n_steps=3)
        for oat_result, param in zip(results, params_spec, strict=True):
            assert oat_result.values[0] == pytest.approx(param.low)
            assert oat_result.values[-1] == pytest.approx(param.high)

    def test_oat_different_values_produce_different_shares(self) -> None:
        config = _fast_config()
        # Pick the first parameter (usually rho or latent_conversion)
        results = run_oat(config, n_steps=3)
        first_param = results[0]
        shares_low = first_param.runs[0].final_shares
        shares_high = first_param.runs[-1].final_shares
        # At least one share should differ meaningfully
        diffs = [
            abs(shares_low[k] - shares_high[k]) for k in shares_low if k in shares_high
        ]
        assert max(diffs) > 1e-6, "Varying parameter had no effect on shares"

    def test_oat_rejects_n_steps_below_2(self) -> None:
        config = _fast_config()
        with pytest.raises(ValueError, match="n_steps must be >= 2"):
            run_oat(config, n_steps=1)

    def test_oat_to_dict_serializable(self) -> None:
        config = _fast_config()
        results = run_oat(config, n_steps=2)
        for oat_result in results:
            d = oat_result.to_dict()
            serialized = json.dumps(d)
            assert isinstance(serialized, str)


class TestLHS:
    """Latin Hypercube Sampling sensitivity analysis."""

    def test_lhs_returns_correct_n_runs(self) -> None:
        config = _fast_config()
        runs, _stats = run_lhs(config, n_samples=5, seed=42)
        assert len(runs) == 5

    def test_lhs_stats_include_per_ideology_shares(self) -> None:
        config = _fast_config()
        _runs, stats = run_lhs(config, n_samples=5, seed=42)
        stat_names = [s.metric_name for s in stats]
        for ideo in config.ideologies:
            expected = f"{ideo.name}_final_share"
            assert expected in stat_names, f"Missing stat for {ideo.name}"

    def test_lhs_stats_include_entropy(self) -> None:
        config = _fast_config()
        _, stats = run_lhs(config, n_samples=5, seed=42)
        stat_names = [s.metric_name for s in stats]
        assert "final_entropy" in stat_names

    def test_lhs_stats_include_health_pass_rate(self) -> None:
        config = _fast_config()
        _, stats = run_lhs(config, n_samples=5, seed=42)
        stat_names = [s.metric_name for s in stats]
        assert "health_pass_rate" in stat_names

    def test_lhs_mean_share_sums_to_about_one(self) -> None:
        config = _fast_config()
        _, stats = run_lhs(config, n_samples=10, seed=42)
        share_stats = [s for s in stats if s.metric_name.endswith("_final_share")]
        mean_total = sum(s.mean for s in share_stats)
        assert abs(mean_total - 1.0) < 0.05, f"Mean shares sum to {mean_total}"

    def test_lhs_reproducible_with_same_seed(self) -> None:
        config = _fast_config()
        runs1, _ = run_lhs(config, n_samples=3, seed=99)
        runs2, _ = run_lhs(config, n_samples=3, seed=99)
        for r1, r2 in zip(runs1, runs2, strict=True):
            assert r1.final_entropy == pytest.approx(r2.final_entropy)

    def test_lhs_rejects_n_samples_below_1(self) -> None:
        config = _fast_config()
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            run_lhs(config, n_samples=0)

    def test_lhs_stats_to_dict_serializable(self) -> None:
        config = _fast_config()
        _, stats = run_lhs(config, n_samples=3, seed=42)
        for s in stats:
            d = s.to_dict()
            serialized = json.dumps(d)
            assert isinstance(serialized, str)


class TestComputeLHSStats:
    """_compute_lhs_stats helper."""

    def test_empty_runs_returns_empty(self) -> None:
        assert _compute_lhs_stats([]) == []


class TestSensitivityResult:
    """Full sensitivity analysis integration."""

    def test_run_sensitivity_oat_only(self) -> None:
        config = _fast_config()
        result = run_sensitivity(
            config,
            oat_steps=2,
            run_oat_analysis=True,
            run_lhs_analysis=False,
        )
        assert len(result.oat) > 0
        assert len(result.lhs_runs) == 0
        assert result.n_parameters > 0

    def test_run_sensitivity_lhs_only(self) -> None:
        config = _fast_config()
        result = run_sensitivity(
            config,
            lhs_samples=3,
            run_oat_analysis=False,
            run_lhs_analysis=True,
            seed=42,
        )
        assert len(result.oat) == 0
        assert len(result.lhs_runs) == 3
        assert len(result.lhs_stats) > 0

    def test_run_sensitivity_both(self) -> None:
        config = _fast_config()
        result = run_sensitivity(
            config,
            oat_steps=2,
            lhs_samples=3,
            seed=42,
        )
        assert len(result.oat) > 0
        assert len(result.lhs_runs) == 3

    def test_full_result_to_dict_serializable(self) -> None:
        config = _fast_config()
        result = run_sensitivity(
            config,
            oat_steps=2,
            lhs_samples=2,
            seed=42,
        )
        d = result.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed["n_parameters"] == result.n_parameters

    def test_metadata_tracks_configuration(self) -> None:
        config = _fast_config()
        result = run_sensitivity(
            config,
            oat_steps=5,
            lhs_samples=7,
            seed=123,
            deterministic=True,
        )
        assert result.metadata["oat_steps"] == 5
        assert result.metadata["lhs_samples"] == 7
        assert result.metadata["seed"] == 123
        assert result.metadata["deterministic"] is True
