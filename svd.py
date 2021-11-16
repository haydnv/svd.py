import itertools

import numpy as np

# from "Numerical Recipes in C" p. 65
EPS = 10**-6


def zeroize(x):
    """Zero out entries `x[i][j]` in the given matrix `x` where `abs(x[i][j]) <= EPS`"""
    return x * (np.abs(x) > EPS)


# based on:
#   https://stackoverflow.com/questions/53489237/how-can-you-implement-householder-based-qr-decomposition-in-python
#   http://drsfenner.org/blog/2016/03/householder-bidiagonalization/
def householder(x):
    """Compute the Householder vector of the given column vector `x`."""

    s = (x[1:]**2).sum()

    if s > EPS:
        alpha = x[0]
        t = np.sqrt(alpha**2 + s)
        v_zero = alpha - t if alpha <= 0 else -s / (alpha + t)
        beta = 2 * v_zero ** 2 / (s + v_zero ** 2)
    else:
        v_zero = 1.
        beta = 0

    v = np.copy(x)
    v[0] = v_zero

    return v / v_zero, beta


# based on http://drsfenner.org/blog/2016/03/householder-bidiagonalization/
def bidiagonalize(x):
    """Compute the bidiagonal form of the matrix `x` using Householder reflections."""

    A = x.copy()
    m, n = A.shape
    assert m >= n
    U, V_t = np.eye(m), np.eye(n)

    def left(k):
        v, beta = householder(A[k:, k])
        A[k:, k:] = (np.eye(m - k) - beta * np.outer(v, v)).dot(A[k:, k:])
        Q = np.eye(m)
        Q[k:, k:] -= beta * np.outer(v, v)
        return U.dot(Q)

    def right(k):
        v, beta = householder(A[k, k + 1:].T)
        A[k:, k + 1:] = A[k:, k + 1:].dot(np.eye(n - (k + 1)) - beta * np.outer(v, v))
        P = np.eye(n)
        P[k + 1:, k + 1:] -= beta * np.outer(v, v)
        return P.dot(V_t)

    for k in range(n):
        U = left(k)

        if k <= n - 2:
            V_t = right(k)

    return U, zeroize(A), V_t


def has_zero_superdiagonal(x):
    """Return `True` if the given square matrix `x` is has any zero entry above its diagonal."""
    assert x.shape[0] == x.shape[1] and x.ndim == 2
    m = x.shape[0]

    for i in range(1, m):
        if not np.diagonal(x[:i, :i]).all():
            return True

    return False


def is_diagonal(x):
    """Return `True` if the given square matrix `x` is diagonal."""
    assert x.shape[0] == x.shape[1] and x.ndim == 2

    return not np.any(x - np.diag(np.diagonal(x)))


# Golub-Reinsch SVD algorithm
# implementation based on:
#   "Numerical Recipes in C" section 2.6, see http://www.grad.hr/nastava/gs/prg/NumericalRecipesinC.pdf
#
#   "Computation of the Singular Value Decomposition",
#   see https://www.cs.utexas.edu/users/inderjit/public_papers/HLA_SVD.pdf
def svd(x):
    """Compute the singular value decomposition of the matrix `x`"""

    m, n = x.shape

    # Golub-Reinsch step 1
    U, B, V_t = bidiagonalize(x)
    V = V_t.T

    # Golub-Reinsch step 2
    while True:

        # Golub-Reinsch step 2a
        for i in range(n - 1):
            if abs(B[i][i + 1]) < EPS:
                B[i][i + 1] = 0

        # Golub-Reinsch step 2b
        B_2_2 = lambda p, q: B[-(q + p):-q, p + 1:-q]
        B_3_3 = lambda q: B[-q:, -q:]

        p = 1
        q = n - 1
        while (p * 2) < n:

            if not is_diagonal(B_3_3(q)):
                break

            if not has_zero_superdiagonal(B_2_2(p, q)):
                break

            p += 1
            q = n - (p * 2)

        # Golub-Reinsch step 2c
        if q == (n - (p + q)):
            Sigma = np.diagonal(B)
            break

        # Golub-Reinsch step 2d
        i_start = p + 1
        i_stop = n - q - 1
        if np.diagonal(B[i_start:i_stop, i_start:i_stop]).any():
            # TODO: apply the Golub-Kahan SVD step
            pass
        else:
            # TODO: Apply Givens rotation so that B[i][i + 1] == 0 and B_2_2(p, q) is still upper bidiagonal
            pass

        break
