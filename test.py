import numpy as np
import unittest
import svd
import anySVD


class SVDTests(unittest.TestCase):
    def testBidiagonalization(self):
        x = np.random.random([10, 4])
        U, A, V = svd.bidiagonalize(x)

        reconstruction = np.matmul(np.matmul(U, A), V.T)
        self.assertTrue(np.allclose(reconstruction, x, rtol=1e-3, atol=1e-3))

    def testSVD(self):
        m, n = 4, 3
        x = np.random.random([m, n])
        U, Sigma, V = svd.svd(x)
        reconstruction = np.matmul(np.matmul(U, Sigma), V.T)

        self.assertTrue(np.allclose(np.matmul(U.T, U), np.eye(n)))
        self.assertTrue(np.allclose(np.matmul(V.T, V), np.eye(n)))
        self.assertTrue(np.allclose(reconstruction, x))

    def testBidiagonalizationAnySVD(self):
        x = np.random.random([4, 80])
        U, A, V = anySVD.bidiagonalize(x)

        reconstruction = np.matmul(np.matmul(U, A), V.T)
        self.assertTrue(np.allclose(reconstruction, x))

    def testAnySVD(self):
        m, n = 4, 80
        x = np.random.random([m, n])
        U, Sigma, V = anySVD.svd(x)

        reconstruction = np.matmul(np.matmul(U, Sigma), V.T)
        self.assertTrue(np.allclose(reconstruction, x, rtol=1e-3, atol=1e-3))


if __name__ == "__main__":
    unittest.main()