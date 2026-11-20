"""Tests for the parameter recovery module (ideasim.evaluation.recovery).

Tests cover:
- draw_truth: drawing parameters within bounds
- synthetic_dataset: generating synthetic data from known configs
- ParameterRecovery: comparison between true and fitted parameters
- compare_to_truth: bulk comparison of fit results with truth
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from ideasim.defaults import make_default_config
from ideasim.evaluation.fitting import (
    FitResult,
    _extract_parameters,
    build_parameter_spec,
    compute_metrics,
    select_parameters,
)
from ideasim.evaluation.recovery import (
    ParameterRecovery,
    compare_to_truth,
    draw_truth,
    synthetic_dataset,
)


def _make_cheap_config() -> tuple[Any, list[float]]:
    """Create a cheap config for testing and return expected years."""
    config = make_default_config()
    config.t_start = 1890.0
    config.dt = 0.5
    years = [1900.0, 1910.0, 1920.0]
    return config, years


# --- TestDrawTruth ---


class TestDrawTruth:
    def test_d1_full_spec_values_within_bounds(self) -> None:
        """D1: every drawn value is within [low, high] for full spec."""
        config, _ = _make_cheap_config()
        spec = build_parameter_spec(config)
        rng = np.random.default_rng(0)
        truth = draw_truth(config, spec, rng)
        values = _extract_parameters(truth, spec)
        for p, v in zip(spec, values, strict=True):
            assert p.low <= v <= p.high

    def test_d2_full_spec_values_within_margin(self) -> None:
        """D2: every value within [low + 0.25w, high - 0.25w] with margin=0.25."""
        config, _ = _make_cheap_config()
        spec = build_parameter_spec(config)
        rng = np.random.default_rng(0)
        truth = draw_truth(config, spec, rng, margin=0.25)
        values = _extract_parameters(truth, spec)
        for p, v in zip(spec, values, strict=True):
            w = p.high - p.low
            assert p.low + 0.25 * w <= v <= p.high - 0.25 * w

    def test_d3_only_rho_others_unchanged(self) -> None:
        """D3: spec with only rho; others equal base config, rho differs."""
        config, _ = _make_cheap_config()
        full_spec = build_parameter_spec(config)
        rho_spec = select_parameters(full_spec, ["rho"])
        rng = np.random.default_rng(0)
        truth = draw_truth(config, rho_spec, rng)
        # Extract with full spec
        truth_values = _extract_parameters(truth, full_spec)
        base_values = _extract_parameters(config, full_spec)
        for p, t_val, b_val in zip(full_spec, truth_values, base_values, strict=True):
            if p.name == "rho":
                assert t_val != b_val
            else:
                assert t_val == b_val

    def test_d4_base_config_unchanged(self) -> None:
        """D4: base config's extracted values unchanged after draw_truth."""
        config, _ = _make_cheap_config()
        spec = build_parameter_spec(config)
        base_values = _extract_parameters(config, spec)
        rng = np.random.default_rng(0)
        _ = draw_truth(config, spec, rng)
        after_values = _extract_parameters(config, spec)
        assert np.array_equal(base_values, after_values)

    def test_d5_deterministic_and_different(self) -> None:
        """D5: rng 7 twice gives identical; rng 7 vs rng 8 gives different."""
        config, _ = _make_cheap_config()
        spec = build_parameter_spec(config)
        rng7a = np.random.default_rng(7)
        rng7b = np.random.default_rng(7)
        rng8 = np.random.default_rng(8)
        truth7a = draw_truth(config, spec, rng7a)
        truth7b = draw_truth(config, spec, rng7b)
        truth8 = draw_truth(config, spec, rng8)
        values7a = _extract_parameters(truth7a, spec)
        values7b = _extract_parameters(truth7b, spec)
        values8 = _extract_parameters(truth8, spec)
        assert np.array_equal(values7a, values7b)
        assert not np.array_equal(values7a, values8)

    def test_d6_invalid_margin_raises(self) -> None:
        """D6: margin -0.1, margin 0.5, empty spec raise ValueError."""
        config, _ = _make_cheap_config()
        spec = build_parameter_spec(config)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            draw_truth(config, spec, rng, margin=-0.1)
        with pytest.raises(ValueError):
            draw_truth(config, spec, rng, margin=0.5)
        with pytest.raises(ValueError):
            draw_truth(config, [], rng, margin=0.0)

    def test_d7_initial_shares_sum_to_one(self) -> None:
        """D7: with include_initial_shares=True, result.initial_shares sums to 1."""
        config, _ = _make_cheap_config()
        spec = build_parameter_spec(config, include_initial_shares=True)
        rng = np.random.default_rng(0)
        truth = draw_truth(config, spec, rng)
        # initial_shares are renormalized
        total = float(np.sum(truth.initial_shares))
        assert total == pytest.approx(1.0)


# --- TestSyntheticDataset ---


class TestSyntheticDataset:
    def test_s1_basic_structure(self) -> None:
        """S1: one observation per (year, ideology); names and years match."""
        config, years = _make_cheap_config()
        dataset = synthetic_dataset(config, years)
        assert len(dataset.observations) == len(years) * len(config.ideologies)
        assert dataset.ideology_names == config.ideology_names()
        assert list(dataset.years) == years

    def test_s2_shares_equal_simulation(self) -> None:
        """S2: noise 0, shares equal the simulation."""
        from ideasim.evaluation.fitting import _simulate_and_extract
        from ideasim.evaluation.recovery import deterministic_copy

        config, years = _make_cheap_config()
        obs_years = np.array(years, dtype=np.float64)
        names = config.ideology_names()
        dataset = synthetic_dataset(config, years)
        sim = deterministic_copy(config)
        sim.t_end = float(obs_years[-1]) + 1.0
        sim_times, sim_shares = _simulate_and_extract(sim, obs_years, names)
        metrics = compute_metrics(
            obs_years,
            dataset.as_matrix(names)[1],
            sim_times,
            sim_shares,
            names,
        )
        assert metrics.mse < 1e-20

    def test_s3_stochastic_params_irrelevant_with_noise_0(self) -> None:
        """S3: stochastic params don't affect result when noise=0."""
        config, years = _make_cheap_config()
        # Config with stochastic params
        config1 = copy.deepcopy(config)
        config1.stochastic.charisma_rate = 5.0
        for ideo in config1.ideologies:
            ideo.mutation_rate = 0.5

        dataset1 = synthetic_dataset(config1, years)
        dataset2 = synthetic_dataset(config, years)

        # Compare shares - they should be identical
        names = config.ideology_names()
        _y1, s1 = dataset1.as_matrix(names)
        _y2, s2 = dataset2.as_matrix(names)
        assert np.array_equal(s1, s2)

    def test_s4_noise_properties(self) -> None:
        """S4: noise_sd 0.05, seed properties."""
        config, _ = _make_cheap_config()
        years = [
            1900.0,
            1910.0,
            1920.0,
            1930.0,
            1940.0,
            1950.0,
            1960.0,
            1970.0,
            1980.0,
            1990.0,
            2000.0,
            2010.0,
            2020.0,
        ]
        names = config.ideology_names()

        # Seed 3 twice
        dataset3a = synthetic_dataset(config, years, noise_sd=0.05, seed=3)
        dataset3b = synthetic_dataset(config, years, noise_sd=0.05, seed=3)
        _y3a, s3a = dataset3a.as_matrix(names)
        _y3b, s3b = dataset3b.as_matrix(names)
        assert np.array_equal(s3a, s3b)

        # Seed 3 vs seed 4
        dataset4 = synthetic_dataset(config, years, noise_sd=0.05, seed=4)
        _y4, s4 = dataset4.as_matrix(names)
        assert not np.array_equal(s3a, s4)

        # Every share in [0, 1]
        assert np.all(s3a >= 0.0)
        assert np.all(s3a <= 1.0)

        # Noise-free dataset
        dataset0 = synthetic_dataset(config, years, noise_sd=0.0)
        _y0, s0 = dataset0.as_matrix(names)

        # Standard deviation of (noisy - noise-free)
        diff = s3a - s0
        std_dev = float(np.std(diff))
        assert 0.02 <= std_dev <= 0.08

    def test_s5_subset_ideologies(self) -> None:
        """S5: only specified ideologies appear."""
        config, years = _make_cheap_config()
        all_names = config.ideology_names()
        subset = all_names[:2]
        dataset = synthetic_dataset(config, years, ideology_names=subset)
        assert dataset.ideology_names == subset
        for obs in dataset.observations:
            assert obs.ideology in subset

    def test_s6_invalid_inputs_raise(self) -> None:
        """S6: various invalid inputs raise ValueError."""
        config, _ = _make_cheap_config()

        with pytest.raises(ValueError):
            synthetic_dataset(config, [])

        with pytest.raises(ValueError):
            synthetic_dataset(config, [1910.0, 1900.0])  # not strictly increasing

        with pytest.raises(ValueError):
            synthetic_dataset(config, [1900.0, 1900.0])  # not strictly increasing

        with pytest.raises(ValueError):
            synthetic_dataset(config, [1900.0], noise_sd=-0.1)

        with pytest.raises(ValueError):
            synthetic_dataset(config, [1900.0], ideology_names=["no_such"])

        with pytest.raises(ValueError):
            synthetic_dataset(config, [1850.0])  # before t_start

    def test_s7_config_unchanged(self) -> None:
        """S7: input config's t_end, seed, charisma_rate unchanged after call."""
        config, years = _make_cheap_config()
        orig_t_end = config.t_end
        orig_seed = config.seed
        orig_charisma = config.stochastic.charisma_rate

        _ = synthetic_dataset(config, years, noise_sd=0.0)

        assert config.t_end == orig_t_end
        assert config.seed == orig_seed
        assert config.stochastic.charisma_rate == orig_charisma

    def test_s8_metadata(self) -> None:
        """S8: metadata contains synthetic, noise_sd, seed."""
        config, years = _make_cheap_config()
        dataset = synthetic_dataset(config, years, noise_sd=0.05, seed=3)
        assert dataset.metadata["synthetic"] is True
        assert dataset.metadata["noise_sd"] == 0.05
        assert dataset.metadata["seed"] == 3


# --- TestParameterRecovery ---


class TestParameterRecovery:
    def test_p1_error_and_scaled_error(self) -> None:
        """P1: error and scaled_error computed correctly."""
        pr = ParameterRecovery(
            name="p",
            true=0.2,
            fitted=0.3,
            low=0.0,
            high=0.5,
        )
        assert pr.error == pytest.approx(0.1)
        assert pr.scaled_error == pytest.approx(0.2)
        pr2 = ParameterRecovery(
            name="q",
            true=0.2,
            fitted=0.3,
            low=0.1,
            high=0.6,
        )
        assert pr2.error == pytest.approx(0.1)
        assert pr2.scaled_error == pytest.approx(0.2)

    def test_p2_at_bound(self) -> None:
        """P2: at_bound detection at various positions."""
        test_cases = [
            (0.0, True),
            (0.1, True),
            (0.00005, True),
            (0.09995, True),
            (0.0002, False),
            (0.0998, False),
            (0.05, False),
        ]
        results = [
            ParameterRecovery(
                name="p", true=0.0, fitted=fitted, low=0.0, high=0.1
            ).at_bound
            for fitted, _ in test_cases
        ]
        expected = [expected for _, expected in test_cases]
        assert results == expected

    def test_p3_to_dict_keys(self) -> None:
        """P3: to_dict has exactly the expected keys with derived values."""
        pr = ParameterRecovery(
            name="p",
            true=0.2,
            fitted=0.3,
            low=0.0,
            high=0.5,
        )
        d = pr.to_dict()
        expected_keys = {
            "name",
            "true",
            "fitted",
            "low",
            "high",
            "error",
            "scaled_error",
            "at_bound",
        }
        assert set(d.keys()) == expected_keys
        assert d["name"] == "p"
        assert d["true"] == 0.2
        assert d["fitted"] == 0.3
        assert d["low"] == 0.0
        assert d["high"] == 0.5
        assert d["error"] == pytest.approx(0.1)
        assert d["scaled_error"] == pytest.approx(0.2)
        assert d["at_bound"] is False


# --- TestCompareToTruth ---


class TestCompareToTruth:
    def test_c1_two_rows_spec_order(self) -> None:
        """C1: compare_to_truth with rho and first spec param; two rows, spec order."""
        config, _ = _make_cheap_config()
        spec = build_parameter_spec(config)
        # Get first parameter name and rho
        first_param = spec[0].name
        free = [first_param, "rho"]
        selected_spec = select_parameters(spec, free)

        rng = np.random.default_rng(0)
        truth = draw_truth(config, selected_spec, rng)

        # Build FitResult by hand
        train_metrics = compute_metrics(
            np.array([1.0]),
            np.array([[0.5]]),
            np.array([1.0]),
            np.array([[0.5]]),
            ["x"],
        )
        fit = FitResult(
            best_config=truth,
            train_metrics=train_metrics,
            parameter_values={first_param: 1.0, "rho": 0.5},  # dummy values
            n_evaluations=1,
            metadata={"include_initial_shares": False, "include_initial_latent": False},
        )

        rows = compare_to_truth(fit, truth)
        assert len(rows) == 2
        # Check order: first_param before rho
        assert rows[0].name == first_param
        assert rows[1].name == "rho"
        # Both errors should be 0 since best_config = truth
        assert rows[0].error == pytest.approx(0.0)
        assert rows[1].error == pytest.approx(0.0)

    def test_c2_empty_metadata_raises(self) -> None:
        """C2: FitResult with empty metadata raises ValueError."""
        config, _ = _make_cheap_config()
        truth = config
        train_metrics = compute_metrics(
            np.array([1.0]),
            np.array([[0.5]]),
            np.array([1.0]),
            np.array([[0.5]]),
            ["x"],
        )
        fit = FitResult(
            best_config=truth,
            train_metrics=train_metrics,
            parameter_values={"rho": 0.5},
            n_evaluations=1,
            metadata={},
        )
        with pytest.raises(ValueError, match="include_initial_shares"):
            compare_to_truth(fit, truth)

    def test_c3_fitted_from_best_config_not_parameter_values(self) -> None:
        """C3: fitted values come from best_config, not parameter_values."""
        config, _ = _make_cheap_config()
        spec = build_parameter_spec(config)
        rho_spec = select_parameters(spec, ["rho"])
        rng = np.random.default_rng(0)
        truth = draw_truth(config, rho_spec, rng)

        # Build FitResult where parameter_values has wrong rho
        train_metrics = compute_metrics(
            np.array([1.0]),
            np.array([[0.5]]),
            np.array([1.0]),
            np.array([[0.5]]),
            ["x"],
        )
        fit = FitResult(
            best_config=truth,
            train_metrics=train_metrics,
            parameter_values={"rho": 999.0},  # wrong value
            n_evaluations=1,
            metadata={"include_initial_shares": False, "include_initial_latent": False},
        )

        rows = compare_to_truth(fit, truth)
        assert len(rows) == 1
        # fitted should equal truth's rho (from best_config), not 999.0
        truth_rho = truth.rho
        assert rows[0].fitted == pytest.approx(truth_rho)

    def test_c4_initial_shares_recovery(self) -> None:
        """C4: compare_to_truth with initial_shares; true == fitted == truth.initial_shares[0]."""
        config, _ = _make_cheap_config()
        spec = build_parameter_spec(config, include_initial_shares=True)
        # Find first initial_share parameter
        first_share_name = None
        for p in spec:
            if p.name.startswith("initial_share."):
                first_share_name = p.name
                break
        assert first_share_name is not None

        selected_spec = select_parameters(spec, [first_share_name])
        rng = np.random.default_rng(0)
        truth = draw_truth(config, selected_spec, rng)

        # Build FitResult
        train_metrics = compute_metrics(
            np.array([1.0]),
            np.array([[0.5]]),
            np.array([1.0]),
            np.array([[0.5]]),
            ["x"],
        )
        fit = FitResult(
            best_config=truth,
            train_metrics=train_metrics,
            parameter_values={first_share_name: 0.5},
            n_evaluations=1,
            metadata={"include_initial_shares": True, "include_initial_latent": False},
        )

        rows = compare_to_truth(fit, truth)
        assert len(rows) == 1
        # The truth value for initial_shares comes from the drawn truth
        # The fitted value should also be from truth (best_config)
        # They should both equal truth.initial_shares[0] (or the appropriate index)
        # Extract the ideology index from the parameter name
        ideo_name = first_share_name.replace("initial_share.", "")
        ideo_idx = config.ideology_names().index(ideo_name)
        expected_share = truth.initial_shares[ideo_idx]
        assert rows[0].true == pytest.approx(expected_share)
        assert rows[0].fitted == pytest.approx(expected_share)

    def test_c5_positive_control(self) -> None:
        """C5: positive control - fitting recovers rho with small error."""
        from ideasim.evaluation.fitting import fit_parameters

        config, _ = _make_cheap_config()
        years = [1900.0, 1910.0, 1920.0]
        spec = build_parameter_spec(config)
        rho_spec = select_parameters(spec, ["rho"])

        # Draw truth
        rng = np.random.default_rng(0)
        truth = draw_truth(config, rho_spec, rng, margin=0.1)

        # Create synthetic dataset
        data = synthetic_dataset(truth, years, seed=1)

        # Fit
        fit = fit_parameters(
            config,
            data,
            free=["rho"],
            max_iterations=10,
            population_size=5,
            seed=3,
        )

        # Compare
        rows = compare_to_truth(fit, truth)
        assert len(rows) == 1
        assert abs(rows[0].scaled_error) < 0.01
        assert rows[0].at_bound is False
