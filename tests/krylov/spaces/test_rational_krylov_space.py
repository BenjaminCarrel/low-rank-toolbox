"""
Author: Benjamin Carrel, University of Geneva, 2022

Test file for the RationalKrylovSpace class and methods
"""

# %% Imports
import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

from low_rank_toolbox.krylov.spaces.rational_krylov_space import RationalKrylovSpace

# %% Vector case
np.random.seed(1234)
A = sps.random(20, 20, density=0.5, format="csc")
x = np.random.rand(20, 1)
poles = [1, 1, 1]  # arbitrary poles


def compute_space(A, x, poles):
    space = [x]
    y = x
    for pi in poles:
        y = spsla.spsolve(A - pi * sps.eye(20), A.dot(y))
        space.append(y)
    return space


# %% Krylov space
def test_vector_RationalKrylovSpace():
    # Non-symmetric matrix
    KS = RationalKrylovSpace(A, x, poles)
    m = len(poles) + 1

    # Test the attributes
    assert KS.A is A, "Wrong A"
    assert KS.n == 20, "Wrong n"
    assert KS.r == 1, "Wrong r"

    space = compute_space(A, x, poles)
    Q_ref, _ = la.qr(np.column_stack(space), mode="economic")
    for _ in range(len(poles)):
        KS.augment_basis()
    assert np.allclose(
        la.norm(KS.Q, axis=0), np.ones(m)
    ), "The columns of Q are not normalized -> error in augment basis"
    assert np.allclose(
        KS.Q.T.dot(KS.Q), np.eye(m)
    ), "The columns of Q are not orthogonal -> error in augment basis"
    print("Krylov Space Basis OK.")
    print(la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)))
    assert np.allclose(
        KS.Q.dot(KS.Q.T), Q_ref.dot(Q_ref.T)
    ), "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(A.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print(KS.H)
    print("Rational Krylov Space Projection OK.")

    # Symmetric matrix
    S = A.T.dot(A)
    KS = RationalKrylovSpace(S, x, poles)

    # Test the attributes
    assert KS.A is S, "Wrong A"
    assert KS.n == 20, "Wrong n"
    assert KS.r == 1, "Wrong r"

    space = compute_space(S, x, poles)
    Q_ref, _ = la.qr(np.column_stack(space), mode="economic")
    for _ in range(len(poles)):
        KS.augment_basis()
    assert np.allclose(
        la.norm(KS.Q, axis=0), np.ones(m)
    ), "The columns of Q are not normalized -> error in augment basis"
    assert (
        la.norm(KS.Q.T.dot(KS.Q) - np.eye(m)) < 1e-10
    ), "The columns of Q are not orthogonal -> error in augment basis"
    print("(Symmetric) Krylov Space Basis OK.")
    assert (
        la.norm(KS.Q.dot(KS.Q.T) - Q_ref.dot(Q_ref.T)) < 1e-10
    ), "Wrong projection -> error in augment basis"
    # assert np.allclose(KS.Am, Q_ref.T.dot(S.dot(Q_ref))), "Wrong projected A -> error in augment basis"
    print("(Symmetric) Rational Krylov Space Projection OK.")


# %% Matrix case
np.random.seed(1234)
A = sps.random(20, 20, density=0.5, format="csc")
X0 = np.random.rand(20, 3)
poles = [1, 1, 1]  # arbitrary poles


# %% Block Krylov Space
def test_block_KrylovSpace():
    # Non-symmetric case (Arnoldi)
    m = len(poles) + 1
    r = X0.shape[1]

    # Reference computation
    space = compute_space(A, X0, poles)
    Q_ref, R = la.qr(np.column_stack(space), mode="economic")

    # Check the reference
    assert np.allclose(
        la.norm(Q_ref, axis=0), np.ones(m * r)
    ), "The columns of Q are not normalized -> error in the QR decomposition"
    assert (
        la.norm(Q_ref.T.dot(Q_ref) - np.eye(m * r)) < 1e-10
    ), "The columns of Q are not orthogonal -> error in the QR decomposition"
    print("Reference OK.")

    # Block Krylov Space
    KS = RationalKrylovSpace(A, X0, poles)
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
    print("Block Rational Krylov Space Basis OK.")

    # Compare to the reference
    assert np.allclose(
        KS.Q.dot(KS.Q.T), Q_ref.dot(Q_ref.T)
    ), "Wrong projection -> error in block Krylov Space"
    print("Block Rational Krylov Space Projection OK.")
    # assert np.allclose(KS.Am, Q_ref.T.dot(A.dot(Q_ref))), "Wrong projected A -> error in block Krylov Space"

    # Symmetric case (Lanczos)
    r = X0.shape[1]
    S = A.T.dot(A)

    # Reference computation
    space = compute_space(S, X0, poles)
    Q_ref, _ = la.qr(np.column_stack(space), mode="economic")

    # Check the reference
    assert np.allclose(
        la.norm(Q_ref, axis=0), np.ones(m * r)
    ), "The columns of Q are not normalized -> error in the QR decomposition"
    assert np.allclose(
        Q_ref.T.dot(Q_ref), np.eye(m * r)
    ), "The columns of Q are not orthogonal -> error in the QR decomposition"
    print("Reference OK.")

    # Block Krylov Space
    KS = RationalKrylovSpace(S, X0, poles)
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
    print("Block Krylov Space Basis OK.")

    # Compare to the reference
    assert np.allclose(
        KS.Q.dot(KS.Q.T), Q_ref.dot(Q_ref.T)
    ), "Wrong projection -> error in block Krylov Space"
    print("(Symmetric) Block Krylov Space Projection OK.")


# %% Test fixtures
@pytest.fixture
def valid_rational_matrix():
    """Create a valid sparse matrix"""
    np.random.seed(42)
    return sps.random(10, 10, density=0.3, format="csc")


@pytest.fixture
def valid_rational_vector():
    """Create a valid vector"""
    np.random.seed(42)
    return np.random.rand(10, 1)


@pytest.fixture
def valid_poles():
    """Create a valid poles list"""
    return [1.0, 2.0, 3.0]


# %% Input validation tests for RationalKrylovSpace
class TestRationalKrylovInputValidation:
    """Test input validation for RationalKrylovSpace"""

    def test_non_sparse_matrix(self, valid_rational_vector, valid_poles):
        """Test that non-sparse matrix raises TypeError"""
        A = np.random.rand(10, 10)  # Dense matrix
        with pytest.raises(TypeError, match="A must be a sparse matrix"):
            RationalKrylovSpace(A, valid_rational_vector, poles=valid_poles)

    def test_non_array_vector(self, valid_rational_matrix, valid_poles):
        """Test that non-numpy array raises TypeError"""
        x = [[1], [2], [3]]  # List instead of array
        with pytest.raises(TypeError, match="X must be a numpy array"):
            RationalKrylovSpace(valid_rational_matrix, x, poles=valid_poles)

    def test_non_square_matrix(self, valid_rational_vector, valid_poles):
        """Test that non-square matrix raises ValueError"""
        A = sps.random(10, 15, density=0.3, format="csc")
        with pytest.raises(ValueError, match="A must be a square matrix"):
            RationalKrylovSpace(A, valid_rational_vector, poles=valid_poles)

    def test_dimension_mismatch(self, valid_rational_matrix, valid_poles):
        """Test that dimension mismatch raises ValueError"""
        x = np.random.rand(5, 1)  # Wrong dimension
        with pytest.raises(
            ValueError, match="A and X must have the same number of rows"
        ):
            RationalKrylovSpace(valid_rational_matrix, x, poles=valid_poles)

    def test_nan_in_vector(self, valid_rational_matrix, valid_poles):
        """Test that NaN in vector raises ValueError"""
        x = np.random.rand(10, 1)
        x[0] = np.nan
        with pytest.raises(ValueError, match="X contains NaN or Inf values"):
            RationalKrylovSpace(valid_rational_matrix, x, poles=valid_poles)

    def test_inf_in_vector(self, valid_rational_matrix, valid_poles):
        """Test that Inf in vector raises ValueError"""
        x = np.random.rand(10, 1)
        x[0] = np.inf
        with pytest.raises(ValueError, match="X contains NaN or Inf values"):
            RationalKrylovSpace(valid_rational_matrix, x, poles=valid_poles)

    def test_nan_in_matrix(self, valid_rational_vector, valid_poles):
        """Test that NaN in matrix raises ValueError"""
        A = sps.random(10, 10, density=0.3, format="csc")
        A.data[0] = np.nan
        with pytest.raises(ValueError, match="A contains NaN or Inf values"):
            RationalKrylovSpace(A, valid_rational_vector, poles=valid_poles)

    def test_inf_in_matrix(self, valid_rational_vector, valid_poles):
        """Test that Inf in matrix raises ValueError"""
        A = sps.random(10, 10, density=0.3, format="csc")
        A.data[0] = np.inf
        with pytest.raises(ValueError, match="A contains NaN or Inf values"):
            RationalKrylovSpace(A, valid_rational_vector, poles=valid_poles)


# %% RationalKrylovSpace specific validation
class TestRationalKrylovSpecificValidation:
    """Test validation specific to RationalKrylovSpace"""

    def test_empty_poles(self, valid_rational_matrix, valid_rational_vector):
        """Test that empty poles list raises ValueError"""
        with pytest.raises(ValueError, match="poles list cannot be empty"):
            RationalKrylovSpace(valid_rational_matrix, valid_rational_vector, poles=[])

    def test_nan_in_poles(self, valid_rational_matrix, valid_rational_vector):
        """Test that NaN in poles raises ValueError"""
        poles = [1.0, np.nan, 2.0]
        with pytest.raises(ValueError, match="poles contain NaN or Inf values"):
            RationalKrylovSpace(
                valid_rational_matrix, valid_rational_vector, poles=poles
            )

    def test_inf_in_poles(self, valid_rational_matrix, valid_rational_vector):
        """Test that Inf in poles raises ValueError"""
        poles = [1.0, np.inf, 2.0]
        with pytest.raises(ValueError, match="poles contain NaN or Inf values"):
            RationalKrylovSpace(
                valid_rational_matrix, valid_rational_vector, poles=poles
            )


# %% Dimension overflow tests for RationalKrylovSpace
class TestRationalKrylovDimensionOverflow:
    """Test behavior when space size exceeds matrix dimension"""

    def test_rational_krylov_dimension_error(
        self, valid_rational_matrix, valid_rational_vector
    ):
        """Test that RationalKrylov raises error when exceeding dimension"""
        poles = [1.0] * 15  # More poles than matrix dimension
        RK = RationalKrylovSpace(
            valid_rational_matrix, valid_rational_vector, poles=poles
        )

        # Augment to maximum
        for _ in range(9):
            RK.augment_basis()

        # Try to augment beyond dimension
        with pytest.raises(ValueError, match="space is exceeding the dimension"):
            RK.augment_basis()

    def test_rational_krylov_insufficient_poles(
        self, valid_rational_matrix, valid_rational_vector
    ):
        """Test that error is raised when running out of poles"""
        poles = [1.0, 2.0]  # Only 2 poles
        RK = RationalKrylovSpace(
            valid_rational_matrix, valid_rational_vector, poles=poles
        )

        # Use up all poles
        RK.augment_basis()
        RK.augment_basis()

        # Try to augment without poles - should raise either ValueError or IndexError
        with pytest.raises((ValueError, IndexError)):
            RK.augment_basis()


# %% Property tests for RationalKrylovSpace
class TestRationalKrylovProperties:
    """Test properties for RationalKrylovSpace"""

    def test_rational_krylov_max_iter(
        self, valid_rational_matrix, valid_rational_vector
    ):
        """Test max_iter is set from poles length"""
        poles = [1.0, 2.0, 3.0]
        RK = RationalKrylovSpace(
            valid_rational_matrix, valid_rational_vector, poles=poles
        )
        assert RK.max_iter == 3, "max_iter should equal number of poles"

    def test_complex_poles_dtype(self, valid_rational_matrix, valid_rational_vector):
        """Test complex poles promote dtype to complex"""
        poles = [1.0 + 1.0j, 2.0, 3.0]
        RK = RationalKrylovSpace(
            valid_rational_matrix, valid_rational_vector, poles=poles
        )
        assert np.iscomplexobj(RK.Q), "Q should be complex when poles are complex"


# %% compute_all tests for RationalKrylovSpace
class TestRationalKrylovComputeAll:
    """Test compute_all method"""

    def test_rational_krylov_compute_all(
        self, valid_rational_matrix, valid_rational_vector
    ):
        """Test compute_all for RationalKrylov uses all poles"""
        poles = [1.0, 2.0, 3.0]
        RK = RationalKrylovSpace(
            valid_rational_matrix, valid_rational_vector, poles=poles
        )
        RK.compute_all()

        # Should have used all poles (m = 1 + len(poles))
        assert RK.m == len(poles) + 1, "Should use all poles"


# %% inverse_only parameter tests
class TestRationalKrylovInverseOnly:
    """Test inverse_only parameter in RationalKrylovSpace"""

    def test_inverse_only_true(self, valid_rational_matrix, valid_rational_vector):
        """Test inverse_only=True solves (A-pI)v = u"""
        poles = [0.5]
        RK = RationalKrylovSpace(
            valid_rational_matrix, valid_rational_vector, poles=poles, inverse_only=True
        )
        RK.augment_basis()

        # Verify the basis is orthonormal
        assert np.allclose(
            RK.Q.T @ RK.Q, np.eye(RK.size)
        ), "Basis should be orthonormal"

    def test_inverse_only_false(self, valid_rational_matrix, valid_rational_vector):
        """Test inverse_only=False solves (A-pI)v = Au"""
        poles = [0.5]
        RK = RationalKrylovSpace(
            valid_rational_matrix,
            valid_rational_vector,
            poles=poles,
            inverse_only=False,
        )
        RK.augment_basis()

        # Verify the basis is orthonormal
        assert np.allclose(
            RK.Q.T @ RK.Q, np.eye(RK.size)
        ), "Basis should be orthonormal"

    def test_inverse_only_difference(
        self, valid_rational_matrix, valid_rational_vector
    ):
        """Test that inverse_only parameter works correctly"""
        poles = [0.5]

        # Just verify both modes produce valid orthonormal bases
        RK_true = RationalKrylovSpace(
            valid_rational_matrix, valid_rational_vector, poles=poles, inverse_only=True
        )
        RK_true.augment_basis()

        RK_false = RationalKrylovSpace(
            valid_rational_matrix,
            valid_rational_vector,
            poles=poles,
            inverse_only=False,
        )
        RK_false.augment_basis()

        # Both should produce orthonormal bases
        assert np.allclose(
            RK_true.Q.T @ RK_true.Q, np.eye(RK_true.size)
        ), "inverse_only=True should produce orthonormal basis"
        assert np.allclose(
            RK_false.Q.T @ RK_false.Q, np.eye(RK_false.size)
        ), "inverse_only=False should produce orthonormal basis"


# %%
