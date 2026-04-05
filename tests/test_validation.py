"""Tests for input validation module."""

from __future__ import annotations

import numpy as np

from ideasim.defaults import make_default_config
from ideasim.validation import ValidationResult, validate_config


class TestValidateDefaultConfig:
    """The default config should always pass validation."""

    def test_default_config_is_valid(self) -> None:
        config = make_default_config()
        result = validate_config(config)
        assert result.is_valid, result.summary()

    def test_default_config_warnings_are_known(self) -> None:
        """Default config may have known warnings (e.g., pre-1800 shocks).

        The default scenario includes French Revolution shocks at 1789,
        before the simulation start of 1800. These are intentionally
        part of the scenario data even though they do not fire.
        """
        config = make_default_config()
        result = validate_config(config)
        # All warnings should be about out-of-range shocks
        for warn in result.warnings:
            assert "outside simulation range" in warn.message, (
                f"Unexpected warning: {warn}"
            )


class TestSharesValidation:
    """Validate initial_shares constraints."""

    def test_shares_must_sum_to_one(self) -> None:
        config = make_default_config()
        config.initial_shares = np.array([0.5, 0.5, 0.5, 0.5])
        result = validate_config(config)
        assert not result.is_valid
        assert any("sum to 1.0" in e.message for e in result.errors)

    def test_shares_must_be_nonnegative(self) -> None:
        config = make_default_config()
        config.initial_shares = np.array([1.5, -0.5, 0.0, 0.0])
        result = validate_config(config)
        assert not result.is_valid
        assert any("negative" in e.message for e in result.errors)

    def test_shares_wrong_length(self) -> None:
        config = make_default_config()
        config.initial_shares = np.array([0.5, 0.5])
        result = validate_config(config)
        assert not result.is_valid
        assert any("Expected length" in e.message for e in result.errors)


class TestLatentValidation:
    def test_latent_wrong_length(self) -> None:
        config = make_default_config()
        config.initial_latent = np.array([0.1])
        result = validate_config(config)
        assert not result.is_valid

    def test_latent_negative(self) -> None:
        config = make_default_config()
        config.initial_latent = np.array([0.1, -0.1, 0.1, 0.1])
        result = validate_config(config)
        assert not result.is_valid


class TestTimeRangeValidation:
    def test_start_after_end(self) -> None:
        config = make_default_config()
        config.t_start = 2025.0
        config.t_end = 1800.0
        result = validate_config(config)
        assert not result.is_valid

    def test_negative_dt(self) -> None:
        config = make_default_config()
        config.dt = -0.1
        result = validate_config(config)
        assert not result.is_valid

    def test_dt_too_large(self) -> None:
        config = make_default_config()
        config.dt = 500.0  # larger than total duration
        result = validate_config(config)
        assert not result.is_valid


class TestMatrixValidation:
    def test_interaction_matrix_wrong_shape(self) -> None:
        config = make_default_config()
        config.interaction_matrix = np.zeros((2, 2))
        result = validate_config(config)
        assert not result.is_valid
        assert any("interaction_matrix" in e.field for e in result.errors)

    def test_suppression_matrix_wrong_shape(self) -> None:
        config = make_default_config()
        config.suppression_matrix = np.zeros((3, 3))
        result = validate_config(config)
        assert not result.is_valid

    def test_stochastic_rates_wrong_shape(self) -> None:
        config = make_default_config()
        config.stochastic.hybridization_rates = np.zeros((2, 2))
        result = validate_config(config)
        assert not result.is_valid


class TestEnvironmentValidation:
    def test_env_initial_wrong_length(self) -> None:
        config = make_default_config()
        config.environment.initial = np.array([1.0])
        result = validate_config(config)
        assert not result.is_valid

    def test_shock_out_of_range_is_warning(self) -> None:
        config = make_default_config()
        config.environment.exogenous_shocks.append((9999.0, 0, 0.1))
        result = validate_config(config)
        # Out of range shock is a warning, not an error
        assert result.is_valid
        assert any("outside simulation range" in w.message for w in result.warnings)


class TestIdeologyParamValidation:
    def test_negative_latent_conversion_is_error(self) -> None:
        config = make_default_config()
        config.ideologies[0].latent_conversion = -0.1
        result = validate_config(config)
        assert not result.is_valid

    def test_extreme_rho_is_warning(self) -> None:
        config = make_default_config()
        config.rho = 0.9
        result = validate_config(config)
        # Still valid but should warn
        assert result.is_valid
        assert any("unusually large" in w.message for w in result.warnings)


class TestValidationResult:
    def test_summary_no_issues(self) -> None:
        result = ValidationResult()
        assert "no issues" in result.summary().lower()

    def test_summary_with_errors(self) -> None:
        result = ValidationResult()
        result.add_error("test", "something broke")
        assert "FAILED" in result.summary()
        assert not result.is_valid
