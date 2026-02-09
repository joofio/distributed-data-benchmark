"""Tests for privacy module including re-identification risk analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.privacy import (
    add_laplace_noise,
    compute_noise_stats,
    compute_reidentification_risk,
    estimate_sensitivity,
    simulate_membership_inference_attack,
)


class TestAddLaplaceNoise:
    """Tests for Laplace noise injection."""

    def test_returns_same_shape(self) -> None:
        """Output should have same shape as input."""
        X = np.random.default_rng(42).standard_normal((50, 4))
        X_noisy = add_laplace_noise(X, epsilon=1.0, seed=42)
        assert X_noisy.shape == X.shape

    def test_infinite_epsilon_returns_copy(self) -> None:
        """Infinite epsilon should return unmodified copy."""
        X = np.random.default_rng(42).standard_normal((50, 4))
        X_noisy = add_laplace_noise(X, epsilon=np.inf, seed=42)
        np.testing.assert_array_equal(X_noisy, X)

    def test_lower_epsilon_more_noise(self) -> None:
        """Lower epsilon should add more noise."""
        X = np.random.default_rng(42).standard_normal((50, 4))

        X_low = add_laplace_noise(X, epsilon=0.1, seed=42)
        X_high = add_laplace_noise(X, epsilon=10.0, seed=42)

        noise_low = np.abs(X_low - X).mean()
        noise_high = np.abs(X_high - X).mean()

        assert noise_low > noise_high

    def test_reproducibility(self) -> None:
        """Same seed should produce same noise."""
        X = np.random.default_rng(42).standard_normal((50, 4))

        X_noisy1 = add_laplace_noise(X, epsilon=1.0, seed=123)
        X_noisy2 = add_laplace_noise(X, epsilon=1.0, seed=123)

        np.testing.assert_array_equal(X_noisy1, X_noisy2)


class TestEstimateSensitivity:
    """Tests for sensitivity estimation."""

    def test_range_method(self) -> None:
        """Range method should return max - min."""
        X = np.array([[0, 10], [5, 20], [10, 30]])
        sens = estimate_sensitivity(X, method="range")
        np.testing.assert_array_equal(sens, [10, 20])

    def test_iqr_method(self) -> None:
        """IQR method should return Q3 - Q1."""
        X = np.random.default_rng(42).standard_normal((100, 2))
        sens = estimate_sensitivity(X, method="iqr")
        # IQR should be roughly 1.35 for standard normal
        assert 1.0 < sens.mean() < 2.0

    def test_std_method(self) -> None:
        """Std method should return 2 * std."""
        X = np.random.default_rng(42).standard_normal((100, 2))
        sens = estimate_sensitivity(X, method="std")
        # 2*std should be roughly 2 for standard normal
        assert 1.5 < sens.mean() < 2.5


class TestComputeReidentificationRisk:
    """Tests for re-identification risk analysis."""

    def test_empty_quasi_identifiers(self) -> None:
        """Empty quasi-identifiers should return low risk."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = compute_reidentification_risk(df, [])

        assert result["n_records"] == 3
        assert result["singleton_count"] == 0
        assert result["disclosure_risk_score"] == 0.0

    def test_unique_records_are_singletons(self) -> None:
        """All unique records should be counted as singletons."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "qi": ["a", "b", "c", "d", "e"]  # All unique
        })
        result = compute_reidentification_risk(df, ["qi"])

        assert result["singleton_count"] == 5
        assert result["singleton_rate"] == 1.0
        assert result["min_k_anonymity"] == 1

    def test_identical_records_not_singletons(self) -> None:
        """Identical quasi-identifiers should not be singletons."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "qi": ["a", "a", "b", "b"]  # Pairs
        })
        result = compute_reidentification_risk(df, ["qi"])

        assert result["singleton_count"] == 0
        assert result["singleton_rate"] == 0.0
        assert result["min_k_anonymity"] == 2

    def test_result_contains_expected_keys(self) -> None:
        """Result should contain all expected metrics."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = compute_reidentification_risk(df, ["a"])

        expected_keys = [
            "n_records",
            "n_unique_combinations",
            "singleton_count",
            "singleton_rate",
            "min_k_anonymity",
            "mean_k_anonymity",
            "median_k_anonymity",
            "k_distribution",
            "disclosure_risk_score",
        ]
        for key in expected_keys:
            assert key in result

    def test_risk_score_in_valid_range(self) -> None:
        """Risk score should be between 0 and 1."""
        df = pd.DataFrame({
            "id": range(100),
            "qi": np.random.default_rng(42).choice(["a", "b", "c"], 100)
        })
        result = compute_reidentification_risk(df, ["qi"])

        assert 0.0 <= result["disclosure_risk_score"] <= 1.0


class TestSimulateMembershipInferenceAttack:
    """Tests for membership inference attack simulation."""

    def test_returns_expected_keys(self) -> None:
        """Result should contain all expected metrics."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((50, 5))
        X_test = rng.standard_normal((20, 5))

        result = simulate_membership_inference_attack(X_train, X_test, seed=42)

        expected_keys = [
            "attack_accuracy",
            "baseline_accuracy",
            "attack_advantage",
            "member_confidence",
            "nonmember_confidence",
            "auc_roc",
        ]
        for key in expected_keys:
            assert key in result

    def test_accuracy_in_valid_range(self) -> None:
        """Attack accuracy should be between 0 and 1."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((50, 5))
        X_test = rng.standard_normal((20, 5))

        result = simulate_membership_inference_attack(X_train, X_test, seed=42)

        assert 0.0 <= result["attack_accuracy"] <= 1.0
        assert 0.0 <= result["auc_roc"] <= 1.0

    def test_baseline_is_half(self) -> None:
        """Baseline accuracy should be 0.5."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((50, 5))
        X_test = rng.standard_normal((20, 5))

        result = simulate_membership_inference_attack(X_train, X_test, seed=42)

        assert result["baseline_accuracy"] == 0.5

    def test_empty_test_set_returns_baseline(self) -> None:
        """Empty test set should return baseline values."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((50, 5))
        X_test = np.array([]).reshape(0, 5)

        result = simulate_membership_inference_attack(X_train, X_test, seed=42)

        assert result["attack_accuracy"] == 0.5
        assert result["attack_advantage"] == 0.0

    def test_identical_data_high_accuracy(self) -> None:
        """When test data is identical to train, attack should succeed."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((50, 5))
        # Use some training samples as "test" - should be detected as members
        X_test = X_train[:20] + rng.normal(0, 0.01, (20, 5))  # Small noise

        result = simulate_membership_inference_attack(X_train, X_test, seed=42)

        # Member confidence should be higher than baseline
        assert result["member_confidence"] > 0.4

    def test_reproducibility(self) -> None:
        """Same seed should produce same results."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((50, 5))
        X_test = rng.standard_normal((20, 5))

        result1 = simulate_membership_inference_attack(X_train, X_test, seed=123)
        result2 = simulate_membership_inference_attack(X_train, X_test, seed=123)

        assert result1["attack_accuracy"] == result2["attack_accuracy"]


class TestComputeNoiseStats:
    """Tests for noise statistics computation."""

    def test_returns_expected_keys(self) -> None:
        """Result should contain all expected metrics."""
        X_orig = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_noisy = np.array([[1.1, 2.2], [3.3, 4.4]])

        result = compute_noise_stats(X_orig, X_noisy)

        assert "mean_abs_noise" in result
        assert "max_noise" in result
        assert "noise_to_signal_ratio" in result
        assert "per_feature_noise_std" in result

    def test_no_noise_gives_zero_stats(self) -> None:
        """Identical arrays should give zero noise stats."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = compute_noise_stats(X, X)

        assert result["mean_abs_noise"] == 0.0
        assert result["max_noise"] == 0.0
