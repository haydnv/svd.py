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

    alpha = x[0]
    s = (x[1:]**2).sum()
    t = np.sqrt(alpha ** 2 + s)

    if s > EPS:
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


# based on https://drsfenner.org/blog/2016/03/givens-rotations-and-qr/
def givens(x, z):
    """Return the Givens matrix to map `x` -> `r` and `z` -> 0, where `r = (x**2 + z**2)**0.5`"""

    r = (x**2 + z**2)**0.5
    c, s = (1., 0) if z == 0 else (x / r, -z / r)
    return np.array([[c, s], [-s, c]])


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


# based on https://www.cs.utexas.edu/users/inderjit/public_papers/HLA_SVD.pdf
def golub_kahan(B, Q, P, p, q):
    """Execute the Golub-Kahan SVD step, modifying `B`, `Q`, and `P` in place."""

    m, n = B.shape

    # step 1
    B_2_2 = B[p + 1:n - q, p + 1:n - q]

    # step 2
    C = np.matmul(B_2_2.T, B_2_2)[-2:, -2:]

    def det(C):
        """Return the determinant of a 2x2 matrix `C`"""
        a, b, c, d = C.flatten().tolist()
        return (a * d) - (b * c)

    # derived using the determinant of a 2x2 matrix and the quadratic formula
    # A = [[a, b], [c, d]]
    # |A - lI| = 0 (where l is an eigenvalue)
    # (a - l)*(d - l) - bc = 0 (the determinant)
    # l**2 - l(a - d) - (ad - bc) = 0 (a quadratic polynomial)
    def eigenvalues(C):
        """Compute the eigenvalues of a 2x2 matrix C"""

        # ax^2 + bx + c = 0
        # so a = 1, b = (C[0][0] - C[1][1]), c = -det(C)
        a = 1
        b = (C[0][0] - C[1][1])
        c = -det(C)

        x1 = (-b + (b**2 - 4*a*c)**0.5) / (2 * a)
        x2 = (-b - (b**2 - 4*a*c)**0.5) / (2 * a)

        return x1, x2

    # step 3
    x1, x2 = eigenvalues(C)

    # assign whichever eigenvalue is closer to C[2][2] to mu
    mu = x1 if abs(C[2][2] - x1) < abs(C[2][2] - x2) else x2

    # step 4
    k = p + 1
    alpha = B[k][k]**2 - mu
    beta = B[k][k] * B[k][k + 1]

    # step 5
    for k in range(p + 1, n - q - 1):
        # step 5a
        rotation = givens(alpha, beta)
        # step 5b
        B[:, [k, k + 1]] = np.matmul(B[:, [k, k + 1]], rotation)
        # step 5c
        P = np.matmul(P, rotation)
        # step 5d
        alpha = B[k][k], beta = B[k + 1][k]
        # step 5e
        rotation = givens(alpha, beta).T
        # step 5f
        B[[k + 1, k], :] = np.matmul(rotation, B[[k + 1, k], :])
        # step 5g
        Q[[k + 1, k], :] = np.matmul(rotation, Q[[k + 1, k], :])
        # step 5h appears to be a no-op


# Golub-Reinsch SVD algorithm
# implementation based on:
#   "Numerical Recipes in C" section 2.6, see http://www.grad.hr/nastava/gs/prg/NumericalRecipesinC.pdf
#
#   "Computation of the Singular Value Decomposition",
#   see https://www.cs.utexas.edu/users/inderjit/public_papers/HLA_SVD.pdf
def svd(x, max_iterations=30):
    """Compute the singular value decomposition of the matrix `x`"""

    m, n = x.shape

    # Golub-Reinsch step 1
    U, B, V_t = bidiagonalize(x)
    V = V_t.T

    # Golub-Reinsch step 2
    num_iterations = 0
    Sigma = None
    while num_iterations < max_iterations:

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
            # apply the Golub-Kahan SVD step
            golub_kahan(B, U, V, p, q)
        else:
            # apply Givens rotation so that B[i][i + 1] == 0 and B_2_2(p, q) is still upper bidiagonal
            rotation = givens(B[i][i + 1], B[i][i])
            B[:, [i + 1, i]] = np.matmul(B[:, [i + 1, i]], rotation)

        num_iterations += 1

    if Sigma is None:
        raise RuntimeError(f"SVD failed to converge in {max_iterations} iterations")
    else:
        return U, Sigma, V
