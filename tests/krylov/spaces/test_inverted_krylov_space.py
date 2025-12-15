"""
Author: Benjamin Carrel, University of Geneva, 2022

Test file for the InvertedKrylovSpace class and methods
"""

# %% Imports
import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

from lowrank.krylov.spaces.inverted_krylov_space import InvertedKrylovSpace

# %% Vector case
np.random.seed(1234)
A = sps.random(20, 20, density=0.5, format="csc")
x = np.random.rand(20, 1)
invA = lambda x: spsla.spsolve(A, x).reshape(x.shape)


def space(inv, x, m):
    def repeat(x, n):
        for i in range(n + 1):
            x = inv(x)
        return x

    return np.column_stack([repeat(x, i) for i in range(m)])


# %% Inverted Krylov space
def test_vector_InvertedKrylovSpace():
    # Non-symmetric matrix
    KS = InvertedKrylovSpace(A, x, invA)

    # Test the attributes
    assert KS.A is A, "Wrong A"
    assert KS.n == 20, "Wrong n"
    assert KS.r == 1, "Wrong r"

    # Test for m = 2
    m = 2
    Q_ref, _ = la.qr(space(invA, x, m), mode="economic")
    KS.augment_basis()
    assert np.allclose(
        la.norm(KS.Q, axis=0), np.ones(2)
    ), "The columns of Q are not normalized -> error in augment basis"
    assert (
        la.norm(KS.Q.T.dot(KS.Q) - np.eye(2)) < 1e-10
    ), "The columns of Q are not orthogonal -> error in augment basis"
    print("Inverted Krylov Space Basis OK.")

    assert (
        la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10
    ), "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(A.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print("Inverted Krylov Space Projection (m=2) OK.")

    # Test for m = 3
    m = 3
    Q_ref, _ = la.qr(space(invA, x, m), mode="economic")
    KS.augment_basis()
    assert np.allclose(
        la.norm(KS.Q, axis=0), np.ones(3)
    ), "The columns of Q are not normalized -> error in augment basis"
    assert (
        la.norm(KS.Q.T.dot(KS.Q) - np.eye(3)) < 1e-10
    ), "The columns of Q are not orthogonal -> error in augment basis"
    assert (
        la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10
    ), "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(A.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print("Inverted Krylov Space Projection (m=3) OK.")

    # Symmetric matrix
    S = A.T.dot(A)
    invS = lambda x: spsla.spsolve(S, x).reshape(x.shape)
    KS = InvertedKrylovSpace(S, x, invS)

    # Test the attributes
    assert KS.A is S, "Wrong A"
    assert KS.n == 20, "Wrong n"
    assert KS.r == 1, "Wrong r"

    # Test for m = 2
    m = 2
    Q_ref, _ = la.qr(space(invS, x, m), mode="economic")
    KS.augment_basis()
    assert np.allclose(
        la.norm(KS.Q, axis=0), np.ones(2)
    ), "The columns of Q are not normalized -> error in augment basis"
    assert (
        la.norm(KS.Q.T.dot(KS.Q) - np.eye(2)) < 1e-10
    ), "The columns of Q are not orthogonal -> error in augment basis"
    print("(Symmetric) Inverted Krylov Space Basis OK.")
    assert (
        la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10
    ), "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(S.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print("(Symmetric) Inverted Krylov Space Projection (m=2) OK.")

    # Test for m = 3
    m = 3
    Q_ref, _ = la.qr(space(invS, x, m), mode="economic")
    KS.augment_basis()
    assert np.allclose(
        la.norm(KS.Q, axis=0), np.ones(3)
    ), "The columns of Q are not normalized -> error in augment basis"
    assert (
        la.norm(KS.Q.T.dot(KS.Q) - np.eye(3)) < 1e-10
    ), "The columns of Q are not orthogonal -> error in augment basis"
    print("(Symmetric) Inverted Krylov Space Basis OK.")
    print("Projection error:" + str(la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T))))
    assert (
        la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10
    ), "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(S.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print("(Symmetric) Inverted Krylov Space Projection (m=3) OK.")


# %% Matrix case
np.random.seed(1234)
A = 10 * sps.random(20, 20, density=0.5, format="csc")
invA = spsla.splu(A).solve
X0 = np.random.rand(20, 3)


# %% Block Krylov Space
def test_block_InvertedKrylovSpace():
    # Non-symmetric case (Arnoldi)
    m = 4
    r = X0.shape[1]

    # Reference computation
    Q_ref, R = la.qr(space(invA, X0, m), mode="economic")

    # Check the reference
    assert np.allclose(
        la.norm(Q_ref, axis=0), np.ones(m * r)
    ), "The columns of Q are not normalized -> error in the QR decomposition"
    assert (
        la.norm(Q_ref.T.dot(Q_ref) - np.eye(m * r)) < 1e-10
    ), "The columns of Q are not orthogonal -> error in the QR decomposition"
    print("Reference OK.")

    # Block Krylov Space
    KS = InvertedKrylovSpace(A, X0, invA)
    for i in range(m - 1):
        KS.augment_basis()

    # Check the basis
    assert KS.Q.shape == (20, m * r), "Wrong shape of Q -> error in block Krylov Space"
    assert KS.Am.shape == (
        m * r,
        m * r,
    ), "Wrong shape of Am -> error in block Krylov Space"
    assert np.allclose(
        la.norm(KS.Q, axis=0), np.ones(m * r)
    ), "The columns of Q are not normalized -> error in block Krylov Space"
    assert (
        la.norm(KS.Q.T.dot(KS.Q) - np.eye(m * r)) < 1e-10
    ), "The columns of Q are not orthogonal -> error in block Krylov Space"
    print("Block Inverted Krylov Space Basis OK.")

    # Compare to the reference
    assert (
        la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10
    ), "Wrong projection -> error in block Krylov Space"
    print("Block Inverted Krylov Space Projection OK.")
    # assert np.allclose(KS.Am, Q_ref.T.dot(A.dot(Q_ref))), "Wrong projected A -> error in block Krylov Space"

    # Symmetric case (Lanczos)
    m = 3
    r = X0.shape[1]
    S = A.T.dot(A)
    invS = lambda x: spsla.spsolve(S, x).reshape(x.shape)

    # Reference computation
    Q_ref, _ = la.qr(space(invS, X0, m), mode="economic")

    # Check the reference
    assert np.allclose(
        la.norm(Q_ref, axis=0), np.ones(m * r)
    ), "The columns of Q are not normalized -> error in the QR decomposition"
    assert (
        la.norm(Q_ref.T.dot(Q_ref) - np.eye(m * r)) < 1e-10
    ), "The columns of Q are not orthogonal -> error in the QR decomposition"
    print("Reference OK.")

    # Block Krylov Space
    KS = InvertedKrylovSpace(S, X0, invS)
    for i in range(m - 1):
        KS.augment_basis()

    # Check the basis
    assert KS.Q.shape == (20, m * r), "Wrong shape of Q -> error in block Krylov Space"
    assert KS.Am.shape == (
        m * r,
        m * r,
    ), "Wrong shape of Am -> error in block Krylov Space"
    assert np.allclose(
        la.norm(KS.Q, axis=0), np.ones(m * r)
    ), "The columns of Q are not normalized -> error in block Krylov Space"
    assert np.allclose(
        KS.Q.T.dot(KS.Q), np.eye(m * r)
    ), "The columns of Q are not orthogonal -> error in block Krylov Space"
    print("Symmetric Block Inverted Krylov Space Basis OK.")

    # Compare to the reference
    print("Projection error:" + str(la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T))))
    assert np.allclose(
        KS.Q.dot(KS.Q.T), Q_ref.dot(Q_ref.T), atol=1e-6
    ), "Wrong projection -> error in block Krylov Space"
    print("Symmetric Block Inverted Krylov Space Projection OK.")


# %% Test fixtures
@pytest.fixture
def valid_inverted_matrix():
    """Create a valid sparse matrix"""
    np.random.seed(42)
    return sps.random(10, 10, density=0.3, format="csc")


@pytest.fixture
def valid_inverted_vector():
    """Create a valid vector"""
    np.random.seed(42)
    return np.random.rand(10, 1)


# %% Input validation tests for InvertedKrylovSpace
class TestInvertedKrylovInputValidation:
    """Test input validation for InvertedKrylovSpace"""

    def test_non_sparse_matrix(self, valid_inverted_vector):
        """Test that non-sparse matrix raises TypeError"""
        A = np.random.rand(10, 10)  # Dense matrix
        with pytest.raises(TypeError, match="A must be a sparse matrix"):
            InvertedKrylovSpace(A, valid_inverted_vector)

    def test_non_array_vector(self, valid_inverted_matrix):
        """Test that non-numpy array raises TypeError"""
        x = [[1], [2], [3]]  # List instead of array
        with pytest.raises(TypeError, match="X must be a numpy array"):
            InvertedKrylovSpace(valid_inverted_matrix, x)

    def test_non_square_matrix(self, valid_inverted_vector):
        """Test that non-square matrix raises ValueError"""
        A = sps.random(10, 15, density=0.3, format="csc")
        with pytest.raises(ValueError, match="A must be a square matrix"):
            InvertedKrylovSpace(A, valid_inverted_vector)

    def test_dimension_mismatch(self, valid_inverted_matrix):
        """Test that dimension mismatch raises ValueError"""
        x = np.random.rand(5, 1)  # Wrong dimension
        with pytest.raises(
            ValueError, match="A and X must have the same number of rows"
        ):
            InvertedKrylovSpace(valid_inverted_matrix, x)

    def test_nan_in_vector(self, valid_inverted_matrix):
        """Test that NaN in vector raises ValueError"""
        x = np.random.rand(10, 1)
        x[0] = np.nan
        with pytest.raises(ValueError, match="X contains NaN or Inf values"):
            InvertedKrylovSpace(valid_inverted_matrix, x)

    def test_inf_in_vector(self, valid_inverted_matrix):
        """Test that Inf in vector raises ValueError"""
        x = np.random.rand(10, 1)
        x[0] = np.inf
        with pytest.raises(ValueError, match="X contains NaN or Inf values"):
            InvertedKrylovSpace(valid_inverted_matrix, x)

    def test_nan_in_matrix(self, valid_inverted_vector):
        """Test that NaN in matrix raises ValueError"""
        A = sps.random(10, 10, density=0.3, format="csc")
        A.data[0] = np.nan
        with pytest.raises(ValueError, match="A contains NaN or Inf values"):
            InvertedKrylovSpace(A, valid_inverted_vector)

    def test_inf_in_matrix(self, valid_inverted_vector):
        """Test that Inf in matrix raises ValueError"""
        A = sps.random(10, 10, density=0.3, format="csc")
        A.data[0] = np.inf
        with pytest.raises(ValueError, match="A contains NaN or Inf values"):
            InvertedKrylovSpace(A, valid_inverted_vector)


# %% InvertedKrylovSpace specific tests
class TestInvertedKrylovSpecific:
    """Test InvertedKrylovSpace specific functionality"""

    def test_singular_matrix_error(self, valid_inverted_vector):
        """Test that singular matrix causes error in InvertedKrylov"""
        # Create singular matrix
        A = sps.csc_matrix(np.zeros((10, 10)))

        # Should raise error during initialization (when computing A^-1)
        with pytest.raises(Exception):  # scipy raises RuntimeError or similar
            InvertedKrylovSpace(A, valid_inverted_vector)

    def test_custom_invA(self, valid_inverted_matrix, valid_inverted_vector):
        """Test that custom invA function works"""
        call_count = [0]

        def custom_inv(x):
            call_count[0] += 1
            import scipy.sparse.linalg as spsla

            return spsla.spsolve(valid_inverted_matrix, x).reshape(x.shape)

        IK = InvertedKrylovSpace(
            valid_inverted_matrix, valid_inverted_vector, invA=custom_inv
        )
        assert call_count[0] >= 1, "Custom invA was not called during initialization"


# %% Advanced functionality tests for InvertedKrylovSpace
class TestInvertedKrylovAdvancedFunctionality:
    """Test convergence and consistency"""

    def test_inverted_krylov_span_consistency(
        self, valid_inverted_matrix, valid_inverted_vector
    ):
        """Test that InvertedKrylov space spans correct subspace"""
        IK = InvertedKrylovSpace(valid_inverted_matrix, valid_inverted_vector)
        for _ in range(2):
            IK.augment_basis()

        # Compute reference - InvertedKrylov starts from A^-1 X
        A_inv = spsla.splu(valid_inverted_matrix).solve
        x_inv = A_inv(valid_inverted_vector.ravel()).reshape(
            valid_inverted_vector.shape
        )
        space = [x_inv]
        for _ in range(2):
            space.append(A_inv(space[-1].ravel()).reshape(valid_inverted_vector.shape))
        Q_ref, _ = la.qr(np.column_stack(space), mode="economic")

        # Check projection equality
        P_computed = IK.Q @ IK.Q.T
        P_ref = Q_ref @ Q_ref.T

        assert np.allclose(
            P_computed, P_ref, atol=1e-8
        ), "InvertedKrylov space should span correct subspace"


# %%
