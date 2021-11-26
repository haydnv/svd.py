import numpy as np
import unittest
import svd


class SVDTests(unittest.TestCase):
    def testBidiagonalization(self):
        x = np.random.random([10, 4])
        U, A, V = svd.bidiagonalize(x)

        reconstruction = np.matmul(np.matmul(U, A), V.T)
        self.assertTrue(np.allclose(reconstruction, x))

    def testSVD(self):
        m, n = 4, 3
        x = np.random.random([m, n])
        U, Sigma, V = svd.svd(x)
        reconstruction = np.matmul(np.matmul(U, Sigma), V.T)

        self.assertTrue(np.allclose(np.matmul(U.T, U), np.eye(n)))
        self.assertTrue(np.allclose(np.matmul(V.T, V), np.eye(n)))
        self.assertTrue(np.allclose(reconstruction, x))


if __name__ == "__main__":
    unittest.main()