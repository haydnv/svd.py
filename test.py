import numpy as np
import unittest

import svd


class SVDTests(unittest.TestCase):
    def testBidiagonalization(self):
        m = np.random.random([4, 3])
        U, A, V_t = svd.bidiagonalize(m)
        reconstruction = np.matmul(np.matmul(U, A), V_t)
        self.assertTrue(np.allclose(reconstruction, m))


if __name__ == "__main__":
    unittest.main()
