"""
Golub-Reinsch SVD algorithm
  This algorithm computes the SVD of any mxn matrix
    - It work for all cases: m>n, m<n and m=n

  This implementation is based on:
  "Numerical Recipes in C" section 2.6, 
  see http://www.grad.hr/nastava/gs/prg/NumericalRecipesinC.pdf

  Author attribution: 
    E.E. Gilberto Galvis

    e-mail          : galvisgilberto@gmail.com | gilbertogalvis2227@gmail.com
    LinkedIn Profile: https://www.linkedin.com/in/gilberto-galvis-73043685/
    Github Profile  : https://github.com/gilbertogalvis
    Upwork Profile  : https://www.upwork.com/freelancers/~011eb9f40bb4fa3208

"""

import numpy as np

# -------------------------- #
#
EPS = 1.e-6  # assumes single-precision

# -------------------------- #
#
# Utility functions
#
"""
Utility functions are based on the header nrutil.h
  To download it please follow:
  https://lweb.cfa.harvard.edu/~sasselov/rec/code/nrutil.h
"""
SQR = lambda a: 0.0 if a == 0.0 else a*a
SIGN = lambda a,b: np.fabs(a) if b>=0.0 else -np.fabs(a)
def PYTHAG(a,b): 
    aa, bb = np.fabs(a), np.fabs(b)
    if aa > bb:
        return aa * np.sqrt(1.0 + SQR(bb/aa))
    else:
        return bb * np.sqrt(1.0 + SQR(aa/bb))

# -------------------------- #
#
# Helper functions
#
def testfsplit(U, W, e, k):
    goto_test_f_convergence = False

    for l in np.arange(k+1)[::-1]:
        if abs(e[l]) <= EPS:
            # goto test f convergence
            goto_test_f_convergence = True
            break  # break out of l loop
        if abs(W[l-1]) <= EPS:
            # goto cancellation
            break  # break out of l loop

    if goto_test_f_convergence: return l

    # cancellation of e[l] if l>0
    cancelation(U, W, e, l, k)
    return l


def cancelation(U, W, e, l, k):
    c, s, l1 = 0.0, 1.0, l-1

    for i in range(l,k+1):
        f, e[i] = s * e[i], c * e[i]

        # abs(f) <= maxiter: goto test f convergence
        if abs(f) <= EPS: break
            
        g = W[i]
        h = PYTHAG(f, g)
        W[i], c, s = h, g/h, -f/h

        Y, Z = U[:,l1].copy(), U[:,i].copy()
        U[:,l1] =  Y * c + Z * s
        U[:,i]  = -Y * s + Z * c


# -------------------------- #
#
def householder(U, W, e):
    """Householder's reduction to bidiagonal form"""

    m, n = U.shape
    g, x = 0.0, 0.0
    scale = 0.0

    for i in range(n):
        e[i], l = scale*g, i+1

        if i < m:
            scale = U[i:,i].dot(U[i:,i])

            if scale <= EPS:
                g = 0.0
            else:
                U[i:,i] /= scale
                s = U[i:,i].dot(U[i:,i])
                f = U[i,i].copy()
                g = -SIGN(np.sqrt(s), f)
                h = f * g - s
                U[i,i] = f - g

                for j in range(l,n):
                    f = U[i:,i].dot(U[i:,j]) / h
                    U[i:,j] += f * U[i:,i]

                U[i:,i] *= scale
        else:
            g = 0.0

        W[i] = scale * g

        if (i < m) and (i != n-1):
            scale = U[i,l:].dot(U[i,l:])

            if scale <= EPS:
                g = 0.0
            else:
                U[i,l:] /= scale
                s = U[i,l:].dot(U[i,l:])
                f = U[i,l].copy()
                g = -SIGN(np.sqrt(s), f)
                h = f * g - s
                U[i,l] = f - g
                e[l:] = U[i,l:] / h

                for j in range(l,m):
                    s = U[j, l:].dot(U[i,l:])
                    U[j,l:] += s * e[l:]

                U[i,l:] *= scale
        else:
            g = 0.0


def rht(U, V, e):
    """Accumulation of right hand transformations (rht)"""

    m, n = U.shape
    g, l = 0.0, 0

    for i in np.arange(n)[::-1]:
        if i < n-1:
            if g != 0.0:
                V[l:,i] = ( U[i,l:] / U[i,l] ) / g
                for j in range(l,n):
                    s = U[i,l:].dot(V[l:,j])
                    V[l:,j] += s * V[l:,i]

            V[i,l:] = V[l:,i] = 0.0

        V[i,i], g, l = 1.0, e[i], i


def lht(U, W):
    """
    Accumulation of left hand transformations (lht)
    """
    m, n = U.shape

    for i in np.arange(min([m,n]))[::-1]:
        l = i+1
        g = W[i]
        U[i,l:] = 0.0

        if g != 0.0:
            # h = U[i,i] * g
            g = 1.0 / g
            for j in range(l,n):
                f = (U[l:,i].dot(U[l:,j]) / U[i,i]) * g
                U[i:,j] += f * U[i:,i]

            U[i:,i] *= g

        else:
            U[i:,i] = 0.0

        U[i,i] += 1.0


def golub_kahan(U, W, V, e, k, maxiter=30):
    """
    Diagonalization of the bidiagonal form: 
        - k is the kth singular value
        - loop over maxiter allowed iteration
    """

    for t in range(maxiter):

        # test f splitting
        l = testfsplit(U, W, e, k)

        # test f convergence
        if l == k:
            # convergence
            if W[k] < 0.0:
                #W[k] is made non-negative
                W[k] *= -1
                V[:,k] *= -1
            break  # break out of iteration loop and move on to next k value

        if t == maxiter-1:
            if __debug__: print ('Error: no convergence.')
            raise ValueError('SVD Error: no convergence found.')

        # shift from bottom 2x2 minor
        x, y, z = W[l], W[k-1], W[k]
        g, h = e[k-1], e[k]

        f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
        g = PYTHAG(f, 1.0)
        f = ((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x

        # next QR transformation
        c = s = 1.0

        for i in range(l+1,k+1):
            g, y = e[i], W[i]
            h, g = s*g, c*g

            z = PYTHAG(f,h)
            e[i-1] = z

            c, s = f/z, h/z
            f, g = x*c+g*s, g*c-x*s
            h, y = y*s, y*c

            X, Z = V[:,i-1].copy(), V[:,i].copy()
            V[:,i-1] = c * X + s * Z
            V[:,i  ] = c * Z - s * X

            z = PYTHAG(f, h)
            W[i-1] = z

            if z >= EPS:
                c, s = f/z, h/z

            f, x = c*g+s*y, c*y-s*g

            Y, Z = U[:,i-1].copy(), U[:,i].copy()
            U[:,i-1] = c * Y + s * Z
            U[:,i  ] = c * Z - s * Y

        e[l], e[k], W[k] = 0.0, f, x

# -------------------------- #
# 
# Main functions
#
def bidiagonalize(A, tosvd=False):
    """
    Reduction to bidiagonal form
    """

    U = np.asarray(A).copy()
    m, n = U.shape

    W = np.zeros(n)
    V = np.zeros((n, n))
    e = np.zeros(n)  # allocate arrays

    householder(U, W, e)
    rht(U, V, e)
    lht(U, W)

    if not tosvd:
        A = np.diag(W)
        for i in range(1, n):
            A[i-1, i] += e[i]

        return U, A, V

    return U, W, V, e


def svd(A, maxiter=30):
    """
    Given a matrix A, this routine computes its SVD A = U.W.VT
      - The matrix U is output mxm matrix. This mean 
        matrix U will have same size of matrix A.
      - The matrix W is ouput as the diagonal mxn matrix that contains 
        the singular values
      - The matrix V (not the transpose VT) is output as nxn matrix V
    """

    # Bidiagonal form
    U, W, V, e = bidiagonalize(A, tosvd=True)

    # Diagonalization of the bidiagonal form: 
    #    - loop over singular values
    #    - for each singular value apply golub-kahan method
    #
    for k in np.arange(U.shape[1])[::-1]:
        golub_kahan(U, W, V, e, k, maxiter=maxiter)

    m, n = U.shape
    idsorted = np.argsort(-W)

    U = U[:,idsorted]
    W = W[idsorted]
    V = V[:,idsorted]

    return U[:,:m], np.diag(W)[:m,:], V
