import numpy as np
import random
import unittest
import svd


MAX_DIM = 1000


class SVDTests(unittest.TestCase):
    def testBidiagonalization(self):
        m, n = random.randint(1, MAX_DIM), random.randint(1, MAX_DIM)
        x = np.random.random([m, n])
        U, A, V = svd.bidiagonalize(x)

        reconstruction = np.matmul(np.matmul(U, A), V.T)
        self.assertTrue(np.allclose(reconstruction, x))

    def testSVD(self):
        m, n = random.randint(1, MAX_DIM), random.randint(1, MAX_DIM)
        x = np.random.random([m, n])
        U, Sigma, V = svd.svd(x)

        reconstruction = np.matmul(np.matmul(U, Sigma), V.T)
        self.assertTrue(np.allclose(reconstruction, x, rtol=1e-3, atol=1e-3))


if __name__ == "__main__":
    for i in range(100):
        unittest.main()
