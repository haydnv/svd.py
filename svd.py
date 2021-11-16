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

    v = np.copy(x)
    s = v[1:].dot(v[1:])
    alpha = v[0]
    v[0] = 1.0

    if s > EPS:
        t = np.sqrt(alpha**2 + s)
        v_zero = alpha - t if alpha <= 0 else -s / (alpha + t)
        v[0] = v_zero
        v = v / v_zero

    beta = 0 if s < EPS else 2 * v_zero**2 / (s + v_zero**2)

    return v, beta


# based on http://drsfenner.org/blog/2016/03/householder-bidiagonalization/
def bidiagonalize(x):
    """Compute the bidiagonal form of the matrix `x` using Householder reflections."""

    A = x.copy()
    m, n = A.shape
    assert m >= n
    U, V_t = np.eye(m), np.eye(n)

    def left(A, k):
        v, beta = householder(A[k:, k])
        A[k:, k:] = (np.eye(m - k) - beta * np.outer(v, v)).dot(A[k:, k:])
        Q = np.eye(m)
        Q[k:, k:] -= beta * np.outer(v, v)
        return U.dot(Q)

    def right(A, k):
        v, beta = householder(A[k, k + 1:].T)
        A[k:, k + 1:] = A[k:, k + 1:].dot(np.eye(n - (k + 1)) - beta * np.outer(v, v))
        P = np.eye(n)
        P[k + 1:, k + 1:] -= beta * np.outer(v, v)
        return P.dot(V_t)

    for k in range(n):
        U = left(A, k)

        if k <= n - 2:
            V_t = right(A, k)

    return U, zeroize(A), V_t
