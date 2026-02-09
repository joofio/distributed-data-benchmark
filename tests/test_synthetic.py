"""Tests for synthetic data generation and ground-truth evaluation."""
from __future__ import annotations

import numpy as np
import pytest

from src.synthetic import (
    evaluate_against_ground_truth,
    generate_config_for_synthetic,
    generate_synthetic_institutions,
)


class TestGenerateSyntheticInstitutions:
    """Tests for synthetic data generation."""

    def test_generates_correct_number_of_institutions(self) -> None:
        """Should generate exactly n_institutions records."""
        for n in [10, 25, 50]:
            data = generate_synthetic_institutions(n_institutions=n, seed=42)
            assert len(data.df) == n
            assert len(data.true_labels) == n

    def test_generates_correct_number_of_clusters(self) -> None:
        """True labels should contain exactly n_clusters unique values."""
        for k in [2, 3, 4, 5]:
            data = generate_synthetic_institutions(
                n_institutions=30, n_clusters=k, seed=42
            )
            assert len(np.unique(data.true_labels)) == k

    def test_labels_in_valid_range(self) -> None:
        """Labels should be 0-indexed integers."""
        data = generate_synthetic_institutions(
            n_institutions=30, n_clusters=4, seed=42
        )
        assert data.true_labels.min() == 0
        assert data.true_labels.max() == 3

    def test_metadata_contains_parameters(self) -> None:
        """Metadata should record generation parameters."""
        data = generate_synthetic_institutions(
            n_institutions=30,
            n_clusters=4,
            cluster_separation=2.5,
            seed=123,
        )
        assert data.metadata["n_institutions"] == 30
        assert data.metadata["n_clusters"] == 4
        assert data.metadata["cluster_separation"] == 2.5
        assert data.metadata["seed"] == 123

    def test_dataframe_has_required_columns(self) -> None:
        """DataFrame should have id, features, and KPI columns."""
        data = generate_synthetic_institutions(
            n_institutions=20,
            n_numeric_features=5,
            n_categorical_features=3,
            seed=42,
        )
        df = data.df

        # Check ID column
        assert "institution_id" in df.columns

        # Check numeric features
        numeric_cols = [c for c in df.columns if c.startswith("numeric_")]
        assert len(numeric_cols) == 5

        # Check categorical features
        cat_cols = [c for c in df.columns if c.startswith("categorical_")]
        assert len(cat_cols) == 3

        # Check KPI columns
        assert "target_rate" in df.columns
        assert "target_std" in df.columns

    def test_reproducibility(self) -> None:
        """Same seed should produce identical data."""
        data1 = generate_synthetic_institutions(n_institutions=20, seed=42)
        data2 = generate_synthetic_institutions(n_institutions=20, seed=42)

        np.testing.assert_array_equal(data1.true_labels, data2.true_labels)
        np.testing.assert_array_almost_equal(
            data1.df["target_rate"].values,
            data2.df["target_rate"].values,
        )

    def test_different_seeds_produce_different_data(self) -> None:
        """Different seeds should produce different data."""
        data1 = generate_synthetic_institutions(n_institutions=20, seed=42)
        data2 = generate_synthetic_institutions(n_institutions=20, seed=43)

        # Labels might be different
        # At minimum, feature values should differ
        assert not np.allclose(
            data1.df["numeric_0"].values,
            data2.df["numeric_0"].values,
        )

    def test_cluster_separation_affects_distances(self) -> None:
        """Higher separation should increase between-cluster distances."""
        low_sep = generate_synthetic_institutions(
            n_institutions=30, n_clusters=3, cluster_separation=0.5, seed=42
        )
        high_sep = generate_synthetic_institutions(
            n_institutions=30, n_clusters=3, cluster_separation=3.0, seed=42
        )

        # Compute mean between-cluster distance for numeric features
        def mean_between_cluster_distance(data):
            numeric_cols = [c for c in data.df.columns if c.startswith("numeric_")]
            X = data.df[numeric_cols].values
            labels = data.true_labels

            distances = []
            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    if labels[i] != labels[j]:
                        distances.append(np.linalg.norm(X[i] - X[j]))
            return np.mean(distances)

        low_dist = mean_between_cluster_distance(low_sep)
        high_dist = mean_between_cluster_distance(high_sep)

        assert high_dist > low_dist


class TestEvaluateAgainstGroundTruth:
    """Tests for ground-truth evaluation metrics."""

    def test_perfect_match_gives_ari_one(self) -> None:
        """Identical labels should give ARI = 1."""
        true_labels = np.array([0, 0, 1, 1, 2, 2])
        predicted = np.array([0, 0, 1, 1, 2, 2])

        result = evaluate_against_ground_truth(predicted, true_labels)
        assert result["ari"] == 1.0
        assert result["nmi"] == 1.0
        assert result["cluster_purity"] == 1.0

    def test_permuted_labels_gives_ari_one(self) -> None:
        """Permuted cluster IDs should still give ARI = 1."""
        true_labels = np.array([0, 0, 1, 1, 2, 2])
        predicted = np.array([2, 2, 0, 0, 1, 1])  # Same structure, different IDs

        result = evaluate_against_ground_truth(predicted, true_labels)
        assert result["ari"] == 1.0

    def test_random_labels_gives_low_ari(self) -> None:
        """Random labels should give ARI near 0."""
        np.random.seed(42)
        true_labels = np.array([0] * 50 + [1] * 50)
        predicted = np.random.randint(0, 2, 100)

        result = evaluate_against_ground_truth(predicted, true_labels)
        assert result["ari"] < 0.3  # Should be near 0

    def test_result_contains_expected_keys(self) -> None:
        """Result dict should have all expected metrics."""
        true_labels = np.array([0, 0, 1, 1])
        predicted = np.array([0, 0, 1, 1])

        result = evaluate_against_ground_truth(predicted, true_labels)

        assert "ari" in result
        assert "nmi" in result
        assert "cluster_purity" in result
        assert "n_predicted_clusters" in result
        assert "n_true_clusters" in result

    def test_cluster_counts_are_correct(self) -> None:
        """Should correctly count number of clusters."""
        true_labels = np.array([0, 0, 1, 1, 2, 2])
        predicted = np.array([0, 0, 1, 1, 1, 1])  # Only 2 predicted clusters

        result = evaluate_against_ground_truth(predicted, true_labels)

        assert result["n_true_clusters"] == 3
        assert result["n_predicted_clusters"] == 2


class TestGenerateConfigForSynthetic:
    """Tests for config generation from synthetic data."""

    def test_config_has_required_sections(self) -> None:
        """Generated config should have all required sections."""
        data = generate_synthetic_institutions(n_institutions=20, seed=42)
        config = generate_config_for_synthetic(data)

        assert "dataset" in config
        assert "features" in config
        assert "targets" in config
        assert "preprocessing" in config
        assert "consensus" in config
        assert "benchmark" in config
        assert "selection" in config
        assert "output" in config

    def test_config_features_match_data(self) -> None:
        """Config features should match generated data columns."""
        data = generate_synthetic_institutions(
            n_institutions=20,
            n_numeric_features=5,
            n_categorical_features=3,
            seed=42,
        )
        config = generate_config_for_synthetic(data)

        numeric_cols = config["features"]["numeric"]
        categorical_cols = config["features"]["categorical"]

        # Should have 5 numeric + n_patients
        assert len([c for c in numeric_cols if c.startswith("numeric_")]) == 5
        assert "n_patients" in numeric_cols

        # Should have 3 categorical
        assert len(categorical_cols) == 3
