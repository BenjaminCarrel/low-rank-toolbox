"""
Tests for the examples provided in the CSSP submodule documentation.

IMPORTANT: ONE TEST = ONE EXAMPLE
=================================
Each test in this file corresponds to exactly one example in the CSSP module
documentation (docstrings, tutorials, etc.). This ensures that:

1. All documented examples are automatically tested and verified to work.
2. If documentation is updated with new examples, corresponding tests MUST be added here.
3. If a test is added or modified here, the corresponding example in the documentation
   MUST be updated to match.

This bidirectional synchronization keeps the documentation accurate and reliable.

Test Organization:
------------------
Each test class corresponds to a specific CSSP algorithm (DEIM, ARP, QDEIM, etc.)
and contains tests for the examples shown in that algorithm's documentation.
"""

import numpy as np
import scipy.linalg as la

from low_rank_toolbox.cssp import (
    ARP,
    DEIM,
    QDEIM,
    Osinsky,
    gpode,
    gpodr,
    oversampling_sQDEIM,
    sQDEIM,
)


class TestDEIM:
    """Tests for DEIM (Discrete Empirical Interpolation Method)."""

    def test_deim_basic(self):
        """Test basic DEIM functionality."""
        # Create orthonormal matrix
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply DEIM
        p = DEIM(U)

        # Verify output
        assert len(p) == k
        assert all(0 <= idx < n for idx in p)
        assert len(set(p)) == k  # All indices should be unique

    def test_deim_with_projector(self):
        """Test DEIM with projector return."""
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply DEIM with projector
        p, P_U = DEIM(U, return_projector=True)

        # Verify outputs
        assert len(p) == k
        assert P_U.shape == (n, k)

        # Verify interpolation property: U ≈ P_U @ U[p, :]
        reconstructed = P_U @ U[p, :]
        error = np.linalg.norm(U - reconstructed, "fro")
        assert error < 1e-10

    def test_deim_with_inverse(self):
        """Test DEIM with inverse return."""
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply DEIM with projector and inverse
        p, P_U, inv_U = DEIM(U, return_projector=True, return_inverse=True)

        # Verify outputs
        assert len(p) == k
        assert P_U.shape == (n, k)
        assert inv_U.shape == (k, k)

        # Verify inverse property
        assert np.allclose(inv_U, np.linalg.inv(U[p, :]))


class TestQDEIM:
    """Tests for QDEIM (QR-based Discrete Empirical Interpolation Method)."""

    def test_qdeim_basic(self):
        """Test basic QDEIM functionality."""
        # Create orthonormal matrix
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply QDEIM
        p = QDEIM(U)

        # Verify output
        assert len(p) == k
        assert all(0 <= idx < n for idx in p)
        assert len(set(p)) == k  # All indices should be unique

    def test_qdeim_with_projector(self):
        """Test QDEIM with projector return."""
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply QDEIM with projector
        p, P_U = QDEIM(U, return_projector=True)

        # Verify outputs
        assert len(p) == k
        assert P_U.shape == (n, k)

        # Verify interpolation property: U ≈ P_U @ U[p, :]
        reconstructed = P_U @ U[p, :]
        error = np.linalg.norm(U - reconstructed, "fro")
        assert error < 1e-10

    def test_qdeim_with_inverse(self):
        """Test QDEIM with inverse return."""
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply QDEIM with projector and inverse
        p, P_U, inv_U = QDEIM(U, return_projector=True, return_inverse=True)

        # Verify outputs
        assert len(p) == k
        assert P_U.shape == (n, k)
        assert inv_U.shape == (k, k)

        # Verify inverse property
        expected_inv = np.linalg.inv(U[p, :])
        assert np.allclose(inv_U, expected_inv)


class TestSQDEIM:
    """Tests for sQDEIM (Strong QDEIM)."""

    def test_sqdeim_basic(self):
        """Test basic sQDEIM functionality."""
        # Create orthonormal matrix
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply sQDEIM
        p = sQDEIM(U)

        # Verify output
        assert len(p) == k
        assert all(0 <= idx < n for idx in p)
        assert len(set(p)) == k  # All indices should be unique

    def test_sqdeim_with_projector(self):
        """Test sQDEIM with projector return."""
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply sQDEIM with projector
        p, P_U = sQDEIM(U, eta=2, return_projector=True)

        # Verify outputs
        assert len(p) == k
        assert P_U.shape == (n, k)

        # Verify interpolation property: U ≈ P_U @ U[p, :]
        reconstructed = P_U @ U[p, :]
        error = np.linalg.norm(U - reconstructed, "fro")
        assert error < 1e-10

    def test_sqdeim_with_inverse(self):
        """Test sQDEIM with inverse return."""
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply sQDEIM with projector and inverse
        p, P_U, inv_U = sQDEIM(U, eta=2, return_projector=True, return_inverse=True)

        # Verify outputs
        assert len(p) == k
        assert P_U.shape == (n, k)
        assert inv_U.shape == (k, k)


class TestARP:
    """Tests for ARP (Adaptive Randomized Pivoting) from documentation examples."""

    def test_arp_real_case(self):
        """Test ARP with real matrix - example from documentation."""
        # Example usage - real case
        np.random.seed(0)
        # Tall matrix with orthonormal columns
        n, r = 10, 4
        U = np.random.randn(n, r)
        U, _ = la.qr(U, mode="economic")

        J = ARP(U, return_projector=False, seed=42)

        # Verify output
        assert len(J) == r
        assert all(0 <= idx < n for idx in J)
        assert len(set(J)) == r  # All indices should be unique

    def test_arp_complex_case(self):
        """Test ARP with complex matrix - example from documentation."""
        # Example usage - complex case
        np.random.seed(0)
        n, r = 10, 4
        U_complex = np.random.randn(n, r) + 1j * np.random.randn(n, r)
        U_complex, _ = la.qr(U_complex, mode="economic")

        J_complex = ARP(U_complex, return_projector=False, seed=42)

        # Verify output
        assert len(J_complex) == r
        assert all(0 <= idx < n for idx in J_complex)
        assert len(set(J_complex)) == r  # All indices should be unique

    def test_arp_with_projector(self):
        """Test ARP with projector return."""
        np.random.seed(42)
        n, r = 100, 10
        U = np.random.randn(n, r)
        U, _ = la.qr(U, mode="economic")

        J, M = ARP(U, return_projector=True, seed=42)

        # Verify outputs
        assert len(J) == r
        assert M.shape == (n, r)

        # Verify interpolation property
        reconstructed = M @ U[J, :]
        error = np.linalg.norm(U - reconstructed, "fro")
        assert error < 1e-10

    def test_arp_with_inverse(self):
        """Test ARP with inverse return."""
        np.random.seed(42)
        n, r = 100, 10
        U = np.random.randn(n, r)
        U, _ = la.qr(U, mode="economic")

        J, M, inv_U = ARP(U, return_projector=True, return_inverse=True, seed=42)

        # Verify outputs
        assert len(J) == r
        assert M.shape == (n, r)
        assert inv_U.shape == (r, r)


class TestOsinsky:
    """Tests for Osinsky's algorithm from documentation examples."""

    def test_osinsky_real_case(self):
        """Test Osinsky with real matrix - example from documentation."""
        # Example usage - real case
        np.random.seed(0)
        # Tall matrix with orthonormal columns
        n, r = 10, 4
        U = np.random.randn(n, r)
        U, _ = la.qr(U, mode="economic")

        J = Osinsky(U, return_projector=False)

        # Verify output
        assert len(J) == r
        assert all(0 <= idx < n for idx in J)
        assert len(set(J)) == r  # All indices should be unique

    def test_osinsky_complex_case(self):
        """Test Osinsky with complex matrix - example from documentation."""
        # Example usage - complex case
        np.random.seed(0)
        n, r = 10, 4
        U_complex = np.random.randn(n, r) + 1j * np.random.randn(n, r)
        U_complex, _ = la.qr(U_complex, mode="economic")

        J_complex = Osinsky(U_complex, return_projector=False)

        # Verify output
        assert len(J_complex) == r
        assert all(0 <= idx < n for idx in J_complex)
        assert len(set(J_complex)) == r  # All indices should be unique

    def test_osinsky_with_projector(self):
        """Test Osinsky with projector return."""
        np.random.seed(42)
        n, r = 100, 10
        U = np.random.randn(n, r)
        U, _ = la.qr(U, mode="economic")

        J, P_U = Osinsky(U, return_projector=True)

        # Verify outputs
        assert len(J) == r
        assert P_U.shape == (n, r)

        # Verify interpolation property
        reconstructed = P_U @ U[J, :]
        error = np.linalg.norm(U - reconstructed, "fro")
        assert error < 1e-10

    def test_osinsky_with_inverse(self):
        """Test Osinsky with inverse return."""
        np.random.seed(42)
        n, r = 100, 10
        U = np.random.randn(n, r)
        U, _ = la.qr(U, mode="economic")

        J, P_U, inv_U = Osinsky(U, return_projector=True, return_inverse=True)

        # Verify outputs
        assert len(J) == r
        assert P_U.shape == (n, r)
        assert inv_U.shape == (r, r)


class TestGPODE:
    """Tests for gpode (Gappy POD with Energy constraint)."""

    def test_gpode_basic(self):
        """Test basic gpode functionality."""
        # Create orthonormal matrix
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply gpode with oversampling
        oversampling_size = 5
        p = gpode(U, oversampling_size=oversampling_size)

        # Verify output
        assert len(p) == k + oversampling_size
        assert all(0 <= idx < n for idx in p)
        assert len(set(p)) == len(p)  # All indices should be unique

    def test_gpode_with_projector(self):
        """Test gpode with projector return."""
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply gpode with projector
        oversampling_size = 5
        m = k + oversampling_size
        p, P_U = gpode(U, oversampling_size=oversampling_size, return_projector=True)

        # Verify outputs
        assert len(p) == m
        # For oversampling with pseudoinverse, P_U has shape (n, m)
        assert P_U.shape == (n, m)

    def test_gpode_with_inverse(self):
        """Test gpode with inverse return."""
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply gpode with projector and inverse
        oversampling_size = 5
        m = k + oversampling_size
        p, P_U, inv_U = gpode(
            U,
            oversampling_size=oversampling_size,
            return_projector=True,
            return_inverse=True,
        )

        # Verify outputs
        assert len(p) == m
        # For oversampling with pseudoinverse, P_U has shape (n, m)
        assert P_U.shape == (n, m)
        # inv_U is U.T.conj() @ P_U, so shape is (k, m)
        assert inv_U.shape == (k, m)


class TestGPODR:
    """Tests for gpodr (Gappy POD with Residual constraint)."""

    def test_gpodr_with_oversampling(self):
        """Test gpodr with oversampling size."""
        # Create orthonormal matrix
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply gpodr with oversampling
        oversampling_size = 5
        p = gpodr(U, oversampling_size=oversampling_size)

        # Verify output
        assert len(p) == k + oversampling_size
        assert all(0 <= idx < n for idx in p)
        assert len(set(p)) == len(p)  # All indices should be unique

    def test_gpodr_with_tolerance(self):
        """Test gpodr with tolerance."""
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply gpodr with tolerance
        p = gpodr(U, tol=10, max_iter=20)

        # Verify output
        assert len(p) >= k  # Should have at least k indices
        assert all(0 <= idx < n for idx in p)
        assert len(set(p)) == len(p)  # All indices should be unique

    def test_gpodr_with_projector(self):
        """Test gpodr with projector return."""
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply gpodr with projector
        oversampling_size = 5
        m = k + oversampling_size
        p, P_U = gpodr(U, oversampling_size=oversampling_size, return_projector=True)

        # Verify outputs
        assert len(p) == m
        # For oversampling with pseudoinverse, P_U has shape (n, m)
        assert P_U.shape == (n, m)

    def test_gpodr_with_inverse(self):
        """Test gpodr with inverse return."""
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply gpodr with projector and inverse
        oversampling_size = 5
        m = k + oversampling_size
        p, P_U, inv_U = gpodr(
            U,
            oversampling_size=oversampling_size,
            return_projector=True,
            return_inverse=True,
        )

        # Verify outputs
        assert len(p) == m
        # For oversampling with pseudoinverse, P_U has shape (n, m)
        assert P_U.shape == (n, m)
        # inv_U is U.T.conj() @ P_U, so shape is (k, m)
        assert inv_U.shape == (k, m)


class TestOversamplingsSQDEIM:
    """Tests for oversampling_sQDEIM."""

    def test_oversampling_sqdeim_basic(self):
        """Test basic oversampling_sQDEIM functionality."""
        # Create orthonormal matrix
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply oversampling_sQDEIM
        oversampling_size = 5
        p = oversampling_sQDEIM(U, oversampling_size=oversampling_size)

        # Verify output
        assert len(p) == k + oversampling_size
        assert all(0 <= idx < n for idx in p)
        assert len(set(p)) == len(p)  # All indices should be unique

    def test_oversampling_sqdeim_with_projection(self):
        """Test oversampling_sQDEIM with projection return."""
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply oversampling_sQDEIM with projection
        oversampling_size = 5
        m = k + oversampling_size
        p, P_U = oversampling_sQDEIM(
            U, oversampling_size=oversampling_size, return_projection=True
        )

        # Verify outputs
        assert len(p) == m
        # For oversampling with pseudoinverse, P_U has shape (n, m)
        assert P_U.shape == (n, m)

    def test_oversampling_sqdeim_with_inverse(self):
        """Test oversampling_sQDEIM with inverse return."""
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Apply oversampling_sQDEIM with projection and inverse
        oversampling_size = 5
        m = k + oversampling_size
        p, P_U, inv_U = oversampling_sQDEIM(
            U,
            oversampling_size=oversampling_size,
            return_projection=True,
            return_inverse=True,
        )

        # Verify outputs
        assert len(p) == m
        # For oversampling with pseudoinverse, P_U has shape (n, m)
        assert P_U.shape == (n, m)
        # inv_U is U.T.conj() @ P_U, so shape is (k, m)
        assert inv_U.shape == (k, m)


class TestCSSPIntegration:
    """Integration tests comparing different CSSP algorithms."""

    def test_all_algorithms_select_valid_indices(self):
        """Test that all CSSP algorithms select valid indices."""
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Test all algorithms that select exactly k indices
        algorithms = [
            ("DEIM", lambda: DEIM(U)),
            ("QDEIM", lambda: QDEIM(U)),
            ("sQDEIM", lambda: sQDEIM(U)),
            ("ARP", lambda: ARP(U, seed=42)),
            ("Osinsky", lambda: Osinsky(U)),
        ]

        for name, algo in algorithms:
            p = algo()
            assert len(p) == k, f"{name} failed: expected {k} indices, got {len(p)}"
            assert all(0 <= idx < n for idx in p), f"{name} failed: invalid indices"
            assert len(set(p)) == k, f"{name} failed: duplicate indices"

    def test_all_algorithms_satisfy_interpolation(self):
        """Test that all CSSP algorithms satisfy interpolation property."""
        n, k = 100, 10
        np.random.seed(42)
        U, _ = np.linalg.qr(np.random.randn(n, k))

        # Test all algorithms with projector
        algorithms = [
            ("DEIM", lambda: DEIM(U, return_projector=True)),
            ("QDEIM", lambda: QDEIM(U, return_projector=True)),
            ("sQDEIM", lambda: sQDEIM(U, return_projector=True)),
            ("ARP", lambda: ARP(U, return_projector=True, seed=42)),
            ("Osinsky", lambda: Osinsky(U, return_projector=True)),
        ]

        for name, algo in algorithms:
            p, P_U = algo()
            reconstructed = P_U @ U[p, :]
            error = np.linalg.norm(U - reconstructed, "fro")
            assert error < 1e-10, f"{name} failed: interpolation error {error:.2e}"
