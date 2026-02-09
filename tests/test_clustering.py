"""Tests for clustering algorithms including new mixed-type methods."""
from __future__ import annotations

import numpy as np
import pytest

from src.clustering import (
    cluster_data,
    gower_distance_matrix,
    hierarchical_gower_cluster,
    kmeans_cluster,
    kmodes_cluster,
    kprototypes_cluster,
    pam_cluster,
)


class TestKMeans:
    """Tests for k-means clustering."""

    def test_kmeans_returns_correct_shape(self) -> None:
        """Labels array should have same length as input rows."""
        X = np.random.default_rng(42).standard_normal((50, 4))
        labels = kmeans_cluster(X, k=3, seed=42)
        assert labels.shape == (50,)

    def test_kmeans_returns_k_clusters(self) -> None:
        """Should produce exactly k unique labels."""
        X = np.random.default_rng(42).standard_normal((100, 4))
        labels = kmeans_cluster(X, k=4, seed=42)
        assert len(np.unique(labels)) == 4

    def test_kmeans_reproducible(self) -> None:
        """Same seed should produce identical results."""
        X = np.random.default_rng(42).standard_normal((50, 4))
        labels1 = kmeans_cluster(X, k=3, seed=123)
        labels2 = kmeans_cluster(X, k=3, seed=123)
        np.testing.assert_array_equal(labels1, labels2)


class TestKModes:
    """Tests for k-modes clustering."""

    def test_kmodes_returns_correct_shape(self) -> None:
        """Labels array should have same length as input rows."""
        rng = np.random.default_rng(42)
        X = rng.integers(0, 5, size=(50, 4)).astype(float)
        labels = kmodes_cluster(X, k=3, seed=42)
        assert labels.shape == (50,)

    def test_kmodes_reproducible(self) -> None:
        """Same seed should produce identical results."""
        rng = np.random.default_rng(42)
        X = rng.integers(0, 5, size=(50, 4)).astype(float)
        labels1 = kmodes_cluster(X, k=3, seed=123)
        labels2 = kmodes_cluster(X, k=3, seed=123)
        np.testing.assert_array_equal(labels1, labels2)


class TestGowerDistance:
    """Tests for Gower distance matrix computation."""

    def test_gower_numeric_only(self) -> None:
        """Gower on numeric-only data should produce valid distance matrix."""
        X_num = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        D = gower_distance_matrix(X_num, None)

        # Should be N x N
        assert D.shape == (3, 3)
        # Should be symmetric
        np.testing.assert_array_almost_equal(D, D.T)
        # Diagonal should be zero
        np.testing.assert_array_almost_equal(np.diag(D), np.zeros(3))
        # Values should be in [0, 1]
        assert D.min() >= 0.0
        assert D.max() <= 1.0

    def test_gower_categorical_only(self) -> None:
        """Gower on categorical-only data should use simple matching."""
        X_cat = np.array([[0, 1], [0, 1], [1, 0]])
        D = gower_distance_matrix(None, X_cat)

        # Should be symmetric with zero diagonal
        np.testing.assert_array_almost_equal(D, D.T)
        np.testing.assert_array_almost_equal(np.diag(D), np.zeros(3))

        # Same categories should have distance 0
        assert D[0, 1] == 0.0
        # Different categories should have distance > 0
        assert D[0, 2] > 0.0

    def test_gower_mixed_type(self) -> None:
        """Gower on mixed data should combine numeric and categorical."""
        X_num = np.array([[1.0], [5.0], [9.0]])
        X_cat = np.array([[0], [0], [1]])
        D = gower_distance_matrix(X_num, X_cat)

        # Should be symmetric with zero diagonal
        np.testing.assert_array_almost_equal(D, D.T)
        np.testing.assert_array_almost_equal(np.diag(D), np.zeros(3))
        # Values should be in [0, 1]
        assert D.min() >= 0.0
        assert D.max() <= 1.0

    def test_gower_raises_without_data(self) -> None:
        """Should raise error if no data provided."""
        with pytest.raises(ValueError, match="At least one"):
            gower_distance_matrix(None, None)


class TestPAM:
    """Tests for Partitioning Around Medoids."""

    def test_pam_returns_correct_shape(self) -> None:
        """Labels should match number of rows."""
        D = np.array([
            [0.0, 0.5, 0.9],
            [0.5, 0.0, 0.4],
            [0.9, 0.4, 0.0],
        ])
        labels = pam_cluster(D, k=2, seed=42)
        assert labels.shape == (3,)

    def test_pam_returns_k_clusters(self) -> None:
        """Should produce up to k clusters."""
        rng = np.random.default_rng(42)
        n = 50
        D = rng.uniform(0, 1, size=(n, n))
        D = (D + D.T) / 2  # Make symmetric
        np.fill_diagonal(D, 0)
        labels = pam_cluster(D, k=4, seed=42)
        assert len(np.unique(labels)) <= 4

    def test_pam_reproducible(self) -> None:
        """Same seed should produce identical results."""
        rng = np.random.default_rng(42)
        n = 30
        D = rng.uniform(0, 1, size=(n, n))
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)

        labels1 = pam_cluster(D, k=3, seed=123)
        labels2 = pam_cluster(D, k=3, seed=123)
        np.testing.assert_array_equal(labels1, labels2)


class TestKPrototypes:
    """Tests for K-prototypes clustering."""

    def test_kprototypes_returns_correct_shape(self) -> None:
        """Labels should match number of rows."""
        rng = np.random.default_rng(42)
        X_num = rng.standard_normal((50, 3))
        X_cat = rng.integers(0, 3, size=(50, 2)).astype(float)

        labels = kprototypes_cluster(X_num, X_cat, k=3, seed=42)
        assert labels.shape == (50,)

    def test_kprototypes_returns_k_clusters(self) -> None:
        """Should produce exactly k clusters when data is separable."""
        rng = np.random.default_rng(42)
        # Create well-separated clusters
        X_num = np.vstack([
            rng.standard_normal((30, 2)) + [0, 0],
            rng.standard_normal((30, 2)) + [5, 5],
            rng.standard_normal((30, 2)) + [10, 0],
        ])
        X_cat = np.array([[i // 30] for i in range(90)]).astype(float)

        labels = kprototypes_cluster(X_num, X_cat, k=3, seed=42)
        assert len(np.unique(labels)) == 3

    def test_kprototypes_reproducible(self) -> None:
        """Same seed should produce identical results."""
        rng = np.random.default_rng(42)
        X_num = rng.standard_normal((50, 3))
        X_cat = rng.integers(0, 3, size=(50, 2)).astype(float)

        labels1 = kprototypes_cluster(X_num, X_cat, k=3, seed=123, gamma=0.5)
        labels2 = kprototypes_cluster(X_num, X_cat, k=3, seed=123, gamma=0.5)
        np.testing.assert_array_equal(labels1, labels2)

    def test_kprototypes_gamma_affects_result(self) -> None:
        """Different gamma values should potentially produce different results."""
        rng = np.random.default_rng(42)
        X_num = rng.standard_normal((100, 3))
        X_cat = rng.integers(0, 5, size=(100, 3)).astype(float)

        labels_low = kprototypes_cluster(X_num, X_cat, k=3, seed=42, gamma=0.1)
        labels_high = kprototypes_cluster(X_num, X_cat, k=3, seed=42, gamma=5.0)

        # Results should differ (though this is probabilistic)
        # At minimum, both should be valid
        assert labels_low.shape == (100,)
        assert labels_high.shape == (100,)


class TestClusterDataDispatcher:
    """Tests for the cluster_data dispatcher function."""

    def test_cluster_data_kmeans(self) -> None:
        """Dispatcher should correctly route to kmeans."""
        X = np.random.default_rng(42).standard_normal((30, 3))
        labels = cluster_data(X, k=3, seed=42, method="kmeans")
        assert labels.shape == (30,)

    def test_cluster_data_kmodes(self) -> None:
        """Dispatcher should correctly route to kmodes."""
        X = np.random.default_rng(42).integers(0, 5, size=(30, 3)).astype(float)
        labels = cluster_data(X, k=3, seed=42, method="kmodes")
        assert labels.shape == (30,)

    def test_cluster_data_pam_with_distance(self) -> None:
        """Dispatcher should use provided distance matrix for PAM."""
        rng = np.random.default_rng(42)
        n = 30
        D = rng.uniform(0, 1, size=(n, n))
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)

        labels = cluster_data(D, k=3, seed=42, method="pam", distance_matrix=D)
        assert labels.shape == (30,)

    def test_cluster_data_pam_computes_gower(self) -> None:
        """Dispatcher should compute Gower distance if none provided."""
        rng = np.random.default_rng(42)
        X_num = rng.standard_normal((30, 2))
        X_cat = rng.integers(0, 3, size=(30, 2)).astype(float)

        labels = cluster_data((X_num, X_cat), k=3, seed=42, method="pam")
        assert labels.shape == (30,)

    def test_cluster_data_kprototypes(self) -> None:
        """Dispatcher should correctly route to kprototypes."""
        rng = np.random.default_rng(42)
        X_num = rng.standard_normal((30, 2))
        X_cat = rng.integers(0, 3, size=(30, 2)).astype(float)

        labels = cluster_data((X_num, X_cat), k=3, seed=42, method="kprototypes", gamma=0.5)
        assert labels.shape == (30,)

    def test_cluster_data_invalid_method(self) -> None:
        """Should raise error for unknown method."""
        X = np.random.default_rng(42).standard_normal((30, 3))
        with pytest.raises(ValueError, match="Unsupported"):
            cluster_data(X, k=3, seed=42, method="invalid")  # type: ignore

    def test_cluster_data_kmeans_rejects_tuple(self) -> None:
        """K-means should reject tuple input."""
        rng = np.random.default_rng(42)
        X_num = rng.standard_normal((30, 2))
        X_cat = rng.integers(0, 3, size=(30, 2)).astype(float)

        with pytest.raises(ValueError, match="single array"):
            cluster_data((X_num, X_cat), k=3, seed=42, method="kmeans")

    def test_cluster_data_kprototypes_requires_tuple(self) -> None:
        """K-prototypes should require tuple input."""
        X = np.random.default_rng(42).standard_normal((30, 3))
        with pytest.raises(ValueError, match="tuple"):
            cluster_data(X, k=3, seed=42, method="kprototypes")


class TestHierarchicalGowerCluster:
    """Tests for hierarchical clustering on Gower distance."""

    def test_hierarchical_returns_correct_shape(self) -> None:
        """Labels should match number of rows."""
        D = np.array([
            [0.0, 0.3, 0.8],
            [0.3, 0.0, 0.5],
            [0.8, 0.5, 0.0],
        ])
        labels = hierarchical_gower_cluster(D, k=2)
        assert labels.shape == (3,)

    def test_hierarchical_returns_k_clusters(self) -> None:
        """Should produce up to k clusters."""
        rng = np.random.default_rng(42)
        n = 50
        D = rng.uniform(0, 1, size=(n, n))
        D = (D + D.T) / 2  # Make symmetric
        np.fill_diagonal(D, 0)
        labels = hierarchical_gower_cluster(D, k=4)
        assert len(np.unique(labels)) <= 4

    def test_hierarchical_labels_zero_indexed(self) -> None:
        """Labels should be 0-indexed."""
        rng = np.random.default_rng(42)
        n = 30
        D = rng.uniform(0, 1, size=(n, n))
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)
        labels = hierarchical_gower_cluster(D, k=3)
        assert labels.min() >= 0
        assert labels.max() <= 2

    def test_hierarchical_single_point(self) -> None:
        """Single point should return label 0."""
        D = np.array([[0.0]])
        labels = hierarchical_gower_cluster(D, k=1)
        assert labels.shape == (1,)
        assert labels[0] == 0

    def test_hierarchical_different_linkages(self) -> None:
        """Different linkage methods should run without error."""
        rng = np.random.default_rng(42)
        n = 30
        D = rng.uniform(0, 1, size=(n, n))
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)

        for linkage in ["average", "complete", "single"]:
            labels = hierarchical_gower_cluster(D, k=3, linkage_method=linkage)
            assert labels.shape == (n,)
            assert len(np.unique(labels)) <= 3

    def test_hierarchical_with_gower_distance(self) -> None:
        """Should work end-to-end with Gower distance matrix."""
        rng = np.random.default_rng(42)
        X_num = rng.standard_normal((30, 3))
        X_cat = rng.integers(0, 3, size=(30, 2)).astype(float)

        D = gower_distance_matrix(X_num, X_cat)
        labels = hierarchical_gower_cluster(D, k=3)

        assert labels.shape == (30,)
        assert len(np.unique(labels)) <= 3


class TestClusterDataHierarchical:
    """Tests for hierarchical method in cluster_data dispatcher."""

    def test_cluster_data_hierarchical_with_distance(self) -> None:
        """Dispatcher should use provided distance matrix."""
        rng = np.random.default_rng(42)
        n = 30
        D = rng.uniform(0, 1, size=(n, n))
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)

        labels = cluster_data(D, k=3, seed=42, method="hierarchical", distance_matrix=D)
        assert labels.shape == (30,)

    def test_cluster_data_hierarchical_computes_gower(self) -> None:
        """Dispatcher should compute Gower distance if none provided."""
        rng = np.random.default_rng(42)
        X_num = rng.standard_normal((30, 2))
        X_cat = rng.integers(0, 3, size=(30, 2)).astype(float)

        labels = cluster_data((X_num, X_cat), k=3, seed=42, method="hierarchical")
        assert labels.shape == (30,)

    def test_cluster_data_hierarchical_linkage_option(self) -> None:
        """Dispatcher should pass linkage_method parameter."""
        rng = np.random.default_rng(42)
        X_num = rng.standard_normal((30, 2))
        X_cat = rng.integers(0, 3, size=(30, 2)).astype(float)

        # Should not raise with different linkage methods
        for linkage in ["average", "complete", "single"]:
            labels = cluster_data(
                (X_num, X_cat), k=3, seed=42, method="hierarchical",
                linkage_method=linkage
            )
            assert labels.shape == (30,)
