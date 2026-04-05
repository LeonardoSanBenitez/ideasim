"""Tests for the evaluation framework.

Tests cover:
- Dataset loading and manipulation
- Temporal splitting (no data leakage)
- Metrics computation
- Parameter fitting (smoke test with tiny iterations)
- End-to-end evaluation runner
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ideasim.defaults import make_default_config
from ideasim.evaluation.dataset import (
    ObservedDataPoint,
    ObservedDataset,
    load_dataset,
    save_dataset,
)
from ideasim.evaluation.metrics import compute_metrics
from ideasim.evaluation.splits import (
    temporal_kfold,
    temporal_train_test_split,
)

# --- Fixtures ---


def _make_test_dataset() -> ObservedDataset:
    """Create a small test dataset with 10 years of data."""
    observations = []
    ideology_names = ["liberalism", "marxism", "traditionalism", "nationalism"]
    # 10 time points, roughly every 20 years
    years = [1850, 1870, 1890, 1910, 1930, 1945, 1960, 1975, 1990, 2010]
    # Shares that roughly sum to 1 at each year
    shares_data = {
        "liberalism": [0.20, 0.30, 0.28, 0.25, 0.15, 0.25, 0.30, 0.32, 0.38, 0.35],
        "marxism": [0.05, 0.08, 0.15, 0.22, 0.25, 0.30, 0.28, 0.30, 0.22, 0.15],
        "traditionalism": [0.55, 0.42, 0.35, 0.28, 0.20, 0.25, 0.27, 0.23, 0.22, 0.22],
        "nationalism": [0.20, 0.20, 0.22, 0.25, 0.40, 0.20, 0.15, 0.15, 0.18, 0.28],
    }

    for t_idx, year in enumerate(years):
        for name in ideology_names:
            observations.append(
                ObservedDataPoint(
                    year=float(year),
                    ideology=name,
                    share=shares_data[name][t_idx],
                )
            )

    return ObservedDataset(
        name="test_dataset",
        description="Test dataset for unit tests",
        ideology_names=ideology_names,
        observations=observations,
    )


# --- Dataset tests ---


class TestObservedDataset:
    def test_years_property(self) -> None:
        ds = _make_test_dataset()
        years = ds.years
        assert len(years) == 10
        assert years[0] == 1850.0
        assert years[-1] == 2010.0
        # Should be sorted
        assert np.all(np.diff(years) > 0)

    def test_year_range(self) -> None:
        ds = _make_test_dataset()
        assert ds.year_range == (1850.0, 2010.0)

    def test_shares_at_year(self) -> None:
        ds = _make_test_dataset()
        shares = ds.shares_at_year(1850.0)
        assert abs(shares["liberalism"] - 0.20) < 1e-10
        assert abs(shares["traditionalism"] - 0.55) < 1e-10

    def test_as_matrix(self) -> None:
        ds = _make_test_dataset()
        sim_names = ["liberalism", "marxism", "traditionalism", "nationalism"]
        years, shares = ds.as_matrix(sim_names)
        assert years.shape == (10,)
        assert shares.shape == (10, 4)
        # First year, liberalism
        assert abs(shares[0, 0] - 0.20) < 1e-10
        # No NaN since all ideologies are present
        assert not np.any(np.isnan(shares))

    def test_as_matrix_missing_ideology(self) -> None:
        ds = _make_test_dataset()
        # Request an ideology that doesn't exist
        _years, shares = ds.as_matrix(["liberalism", "anarchism"])
        assert shares.shape == (10, 2)
        assert not np.any(np.isnan(shares[:, 0]))  # liberalism present
        assert np.all(np.isnan(shares[:, 1]))  # anarchism missing

    def test_filter_years(self) -> None:
        ds = _make_test_dataset()
        filtered = ds.filter_years(1900.0, 1960.0)
        years = filtered.years
        assert all(1900.0 <= y <= 1960.0 for y in years)
        # 1910, 1930, 1945, 1960
        assert len(years) == 4

    def test_ideology_mapping(self) -> None:
        ds = ObservedDataset(
            name="test",
            description="",
            ideology_names=["social_democracy"],
            observations=[
                ObservedDataPoint(year=2000.0, ideology="social_democracy", share=0.3),
            ],
            ideology_mapping={"social_democracy": "marxism"},
        )
        shares = ds.shares_at_year(2000.0)
        assert "marxism" in shares
        assert abs(shares["marxism"] - 0.3) < 1e-10


class TestDatasetIO:
    def test_save_and_load_roundtrip(self) -> None:
        ds = _make_test_dataset()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        save_dataset(ds, path)
        loaded = load_dataset(path)
        assert loaded.name == ds.name
        assert len(loaded.observations) == len(ds.observations)
        assert loaded.ideology_names == ds.ideology_names


# --- Splits tests ---


class TestTemporalSplits:
    def test_single_split_70_30(self) -> None:
        ds = _make_test_dataset()
        split = temporal_train_test_split(ds, train_fraction=0.7)
        # 10 years, 70% -> 7 train years, 3 val years
        train, val = split.split_dataset(ds)
        train_years = set(train.years)
        val_years = set(val.years)
        # No overlap
        assert len(train_years & val_years) == 0
        # Temporal ordering: all train years < all val years
        assert max(train_years) < min(val_years)

    def test_kfold_no_leakage(self) -> None:
        ds = _make_test_dataset()
        folds = temporal_kfold(ds, n_folds=3, min_train_fraction=0.3)
        assert len(folds) == 3

        for fold in folds:
            train, val = fold.split_dataset(ds)
            train_years = set(train.years)
            val_years = set(val.years)
            # No overlap
            assert len(train_years & val_years) == 0, (
                f"Fold {fold.fold_index}: overlap between train and val"
            )
            # Temporal ordering
            if len(train_years) > 0 and len(val_years) > 0:
                assert max(train_years) < min(val_years), (
                    f"Fold {fold.fold_index}: train year {max(train_years)} "
                    f">= val year {min(val_years)}"
                )

    def test_kfold_expanding_window(self) -> None:
        ds = _make_test_dataset()
        folds = temporal_kfold(ds, n_folds=3, min_train_fraction=0.3)
        # Each subsequent fold should have more training data
        for i in range(len(folds) - 1):
            assert folds[i].train_end <= folds[i + 1].train_end

    def test_kfold_too_few_years(self) -> None:
        ds = ObservedDataset(
            name="tiny",
            description="",
            ideology_names=["a"],
            observations=[
                ObservedDataPoint(year=2000.0, ideology="a", share=0.5),
                ObservedDataPoint(year=2001.0, ideology="a", share=0.5),
            ],
        )
        with pytest.raises(ValueError):
            temporal_kfold(ds, n_folds=5)


# --- Metrics tests ---


class TestMetrics:
    def test_perfect_prediction(self) -> None:
        years = np.array([1900.0, 1950.0, 2000.0])
        shares = np.array(
            [
                [0.3, 0.7],
                [0.4, 0.6],
                [0.5, 0.5],
            ]
        )
        metrics = compute_metrics(
            observed_years=years,
            observed_shares=shares,
            simulated_years=years,
            simulated_shares=shares,
            ideology_names=["a", "b"],
        )
        assert metrics.mse < 1e-10
        assert metrics.mae < 1e-10
        assert metrics.r_squared > 0.999

    def test_known_error(self) -> None:
        years = np.array([2000.0])
        observed = np.array([[0.5, 0.5]])
        predicted = np.array([[0.6, 0.4]])
        metrics = compute_metrics(
            observed_years=years,
            observed_shares=observed,
            simulated_years=years,
            simulated_shares=predicted,
            ideology_names=["a", "b"],
        )
        # Error is 0.1 for each, MSE = 0.01
        assert abs(metrics.mse - 0.01) < 1e-10
        assert abs(metrics.mae - 0.1) < 1e-10

    def test_handles_nan(self) -> None:
        years = np.array([2000.0])
        observed = np.array([[0.5, np.nan]])
        predicted = np.array([[0.6, 0.4]])
        metrics = compute_metrics(
            observed_years=years,
            observed_shares=observed,
            simulated_years=years,
            simulated_shares=predicted,
            ideology_names=["a", "b"],
        )
        # Only 1 valid observation
        assert metrics.n_observations == 1
        assert abs(metrics.mse - 0.01) < 1e-10

    def test_interpolation(self) -> None:
        """Simulated data at different times should be interpolated."""
        obs_years = np.array([1950.0])
        obs_shares = np.array([[0.5, 0.5]])
        # Simulation has data at 1900 and 2000
        sim_years = np.array([1900.0, 2000.0])
        sim_shares = np.array(
            [
                [0.4, 0.6],
                [0.6, 0.4],
            ]
        )
        metrics = compute_metrics(
            observed_years=obs_years,
            observed_shares=obs_shares,
            simulated_years=sim_years,
            simulated_shares=sim_shares,
            ideology_names=["a", "b"],
        )
        # Interpolated at 1950: [0.5, 0.5] -> perfect match
        assert metrics.mse < 1e-10

    def test_to_dict_serializable(self) -> None:
        years = np.array([2000.0])
        shares = np.array([[0.5, 0.5]])
        metrics = compute_metrics(
            observed_years=years,
            observed_shares=shares,
            simulated_years=years,
            simulated_shares=shares,
            ideology_names=["a", "b"],
        )
        d = metrics.to_dict()
        # Should be JSON-serializable
        json.dumps(d)


# --- Fitting smoke test ---


class TestFitting:
    @pytest.mark.slow
    def test_smoke_fit(self) -> None:
        """Smoke test: fitting runs without crashing with minimal iterations.

        Uses a coarse dt and short time range for speed.
        """
        from ideasim.evaluation.fitting import fit_parameters

        ds = _make_test_dataset()
        # Use only a small subset for speed
        train = ds.filter_years(1850.0, 1890.0)
        config = make_default_config()
        # Coarsen for speed
        config.dt = 0.5
        config.record_interval = 2

        result = fit_parameters(
            base_config=config,
            train_dataset=train,
            max_iterations=1,
            population_size=3,
            seed=42,
        )

        assert result.best_config is not None
        assert result.train_metrics.n_observations > 0
        assert len(result.parameter_values) > 0
        assert result.n_evaluations > 0


# --- Runner smoke test ---


class TestRunner:
    @pytest.mark.slow
    def test_smoke_single_split(self) -> None:
        """Smoke test: full evaluation with single split, minimal iterations.

        Uses a coarse dt and short time range for speed.
        """
        from ideasim.evaluation.runner import run_evaluation

        ds = _make_test_dataset()
        config = make_default_config()
        # Coarsen for speed
        config.dt = 0.5
        config.record_interval = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_evaluation(
                base_config=config,
                dataset=ds,
                n_folds=1,
                train_fraction=0.7,
                output_dir=tmpdir,
                max_iterations=1,
                population_size=3,
                seed=42,
            )

            assert len(result.folds) == 1
            assert result.aggregate_val_metrics.get("n_valid_folds", 0) == 1

            # Check files were saved
            assert (Path(tmpdir) / "evaluation_results.json").exists()
            assert (Path(tmpdir) / "fitted_config.json").exists()

            # Verify JSON is valid
            with open(Path(tmpdir) / "evaluation_results.json") as f:
                data = json.load(f)
            assert "folds" in data
            assert "aggregate_val_metrics" in data
