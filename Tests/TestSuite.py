import unittest
from HPC import HPC
import numpy as np


class kWTAHPC(unittest.TestCase):

    def test_sum_is_k(self):
        hpc = HPC([49, 240, 1600, 480, 49],
          0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
          0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
          0.7, 1, 0.1, 0.5,  # gamma, epsilon, nu, turnover rate
          0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha
        I = np.asarray([[1, -1, 1, -1, 1, -1, 1] * 7], dtype=np.float32)
        result = hpc.kWTA(I, 0.2)
        k = np.round(I.shape[1] * 0.2)
        self.assertEqual(sum(result[0]), k)


if __name__ == '__main__':
    unittest.main()
