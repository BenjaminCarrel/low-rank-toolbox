"""
Strong rank-revealing QR factorization.

Implementation from the MATLAB code available at:
https://fr.mathworks.com/matlabcentral/fileexchange/69139-strong-rank-revealing-qr-decomposition

Reference:
Gu, Ming, and Stanley C. Eisenstat. "Efficient algorithms for computing a strong rank-revealing QR factorization." SIAM Journal on Scientific Computing 17.4 (1996): 848-869.

Converted to Python with ChatGPT.
"""


import numpy as np
from scipy.linalg import qr, solve_triangular

def givens(x, y):
    if y == 0:
        c, s = 1, 0
    else:
        if abs(y) >= abs(x):
            cotangent = x / y
            s = 1 / np.sqrt(1 + cotangent**2)
            c = s * cotangent
        else:
            tangent = y / x
            c = 1 / np.sqrt(1 + tangent**2)
            s = c * tangent
    
    return np.array([[c, s], [-s, c]])

def sRRQR_rank(A, f, k):
    if f < 1:
        print("parameter f given is less than 1. Automatically set f = 2")
        f = 2
    
    m, n = A.shape
    k = min(k, m, n)
    Q, R, p = qr(A, pivoting=True)
    
    if k == n:
        return Q[:, :k], R[:k, :], p
    
    R = np.sign(np.diag(R))[:, None] * R
    Q = Q * np.sign(np.diag(R))
    
    AB = solve_triangular(R[:k, :k], R[:k, k:])
    gamma = np.linalg.norm(R[k:, k:], axis=0) if k < R.shape[0] else np.zeros(n-k)
    tmp = solve_triangular(R[:k, :k], np.eye(k))
    omega = 1 / np.sqrt(np.sum(tmp**2, axis=1))
    
    while True:
        tmp = (1 / omega[:, None] * gamma[None, :])**2 + AB**2
        idx = np.argwhere(tmp > f*f)
        if idx.size == 0:
            break
        i, j = idx[0]
        
        AB[:, [0, j]] = AB[:, [j, 0]]
        gamma[[0, j]] = gamma[[j, 0]]
        R[:, [k, k+j]] = R[:, [k+j, k]]
        p[[k, k+j]] = p[[k+j, k]]
        
    return Q[:, :k], R[:k, :], p

def sRRQR_tol(A, f, tol):
    if f < 1:
        print("parameter f given is less than 1. Automatically set f = 2")
        f = 2
    
    m, n = A.shape
    Q, R, p = qr(A, pivoting=True)
    p = np.argsort(p)
    
    R = np.sign(np.diag(R))[:, None] * R
    Q = Q * np.sign(np.diag(R))
    
    k = np.where(np.diag(R) > tol)[0]
    if k.size == 0:
        return np.zeros((m, 0)), np.zeros((0, n)), np.arange(n)
    k = k[-1] + 1
    
    return Q[:, :k], R[:k, :], p

def sRRQR(A, f, mode, param):
    if mode == "rank":
        return sRRQR_rank(A, f, param)
    elif mode == "tol":
        return sRRQR_tol(A, f, param)
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'rank' or 'tol'.")

if __name__ == "__main__":
    X0 = 4 * (1 - 2 * np.random.rand(200, 3))
    Y0 = 4 * (1 - 2 * np.random.rand(200, 3)) + np.array([12, 0, 0])
    
    dist = np.linalg.norm(X0[:, None, :] - Y0[None, :, :], axis=-1)
    A = 1.0 / dist
    
    k = 20
    f = 1.05
    Q, R, p = sRRQR(A, f, "rank", k)
    error = np.linalg.norm(A[:, p] - Q @ R, 'fro') / np.linalg.norm(A, 'fro')
    entry = np.max(np.abs(np.linalg.inv(R[:k, :k]) @ R[:k, k:]))
    
    print(f"Relative approx. error: {error:.3E}")
    print(f"Maximum entry in inv(R11)*R12: {entry:.3f}\n")
    
    tol = 1e-4
    f = 1.01
    Q, R, p = sRRQR(A, f, "tol", tol)
    error = np.linalg.norm(A[:, p] - Q @ R, 'fro') / np.linalg.norm(A, 'fro')
    k = R.shape[0]
    entry = np.max(np.abs(np.linalg.inv(R[:k, :k]) @ R[:k, k:]))
    
    print(f"Approximation rank: {k}")
    print(f"Relative approx. error: {error:.3E}")
    print(f"Maximum entry in inv(R11)*R12: {entry:.3f}\n")
