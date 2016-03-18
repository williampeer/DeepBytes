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


class TestColumnWeigthsUpdate(unittest.TestCase):

    def test_all_columns_sum_ec_ca3(self):
        hpc = HPC([49, 240, 1600, 480, 49],
          0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
          0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
          0.7, 1, 0.1, 0.5,  # gamma, epsilon, nu, turnover rate
          0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha

        no_of_columns = hpc.ec_ca3_weights.get_value().shape[1]

        empty_array = np.zeros(1, hpc.ec_ca3_weights.shape[0]).astype(np.float32)
        for column_index in range(no_of_columns):
            hpc.update_ec_ca3_weights_column(column_index, empty_array)

        matrix_sum = np.sum(hpc.ec_ca3_weights.get_value())
        self.assertEqual(matrix_sum, 0, msg="Sum was not zero after all elements set to zero. Sum: "+str(matrix_sum))

        full_array = np.ones_like(empty_array).astype(np.float32)

        # set one column only:
        num_of_column_elements = hpc.ec_ca3_weights.get_value().shape[0]
        hpc.update_ec_ca3_weights_column(5, full_array)
        self.assertEqual(np.sum(hpc.ec_ca3_weights.get_value()), num_of_column_elements,
                         msg="Matrix sum not equal to the number of column elements.")

        for column_index in range(no_of_columns):
            hpc.update_ec_ca3_weights_column(column_index, full_array)

        matrix_sum = np.sum(hpc.ec_ca3_weights.get_value())
        self.assertEqual(matrix_sum, no_of_columns*hpc.ec_ca3_weights.get_value().shape[0], msg="Sum not cols*rows.")

    def test_all_columns_sum_ec_dg(self):
        hpc = HPC([49, 240, 1600, 480, 49],
          0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
          0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
          0.7, 1, 0.1, 0.5,  # gamma, epsilon, nu, turnover rate
          0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha

        no_of_columns = hpc.ec_dg_weights.get_value().shape[1]

        empty_array = np.zeros(1, hpc.ec_dg_weights.shape[0]).astype(np.float32)
        for column_index in range(no_of_columns):
            hpc.update_ec_dg_weights_column(column_index, empty_array)

        matrix_sum = np.sum(hpc.ec_dg_weights.get_value())
        self.assertEqual(matrix_sum, 0, msg="Sum was not zero after all elements set to zero. Sum: "+str(matrix_sum))

        full_array = np.ones_like(empty_array).astype(np.float32)

        # set one column only:
        num_of_column_elements = hpc.ec_dg_weights.get_value().shape[0]
        hpc.update_ec_dg_weights_column(5, full_array)
        self.assertEqual(np.sum(hpc.ec_dg_weights.get_value()), num_of_column_elements,
                         msg="Matrix sum not equal to the number of column elements.")

        for column_index in range(no_of_columns):
            hpc.update_ec_dg_weights_column(column_index, full_array)

        matrix_sum = np.sum(hpc.ec_dg_weights.get_value())
        self.assertEqual(matrix_sum, no_of_columns*hpc.ec_dg_weights.get_value().shape[0], msg="Sum not cols*rows.")

    def test_all_columns_sum_dg_ca3(self):
        hpc = HPC([49, 240, 1600, 480, 49],
          0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
          0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
          0.7, 1, 0.1, 0.5,  # gamma, epsilon, nu, turnover rate
          0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha

        no_of_columns = hpc.dg_ca3_weights.get_value().shape[1]

        empty_array = np.zeros(1, hpc.dg_ca3_weights.shape[0]).astype(np.float32)
        for column_index in range(no_of_columns):
            hpc.update_dg_ca3_weights_column(column_index, empty_array)

        matrix_sum = np.sum(hpc.dg_ca3_weights.get_value())
        self.assertEqual(matrix_sum, 0, msg="Sum was not zero after all elements set to zero. Sum: "+str(matrix_sum))

        full_array = np.ones_like(empty_array).astype(np.float32)

        # set one column only:
        num_of_column_elements = hpc.dg_ca3_weights.get_value().shape[0]
        hpc.update_dg_ca3_weights_column(5, full_array)
        self.assertEqual(np.sum(hpc.dg_ca3_weights.get_value()), num_of_column_elements,
                         msg="Matrix sum not equal to the number of column elements.")

        for column_index in range(no_of_columns):
            hpc.update_dg_ca3_weights_column(column_index, full_array)

        matrix_sum = np.sum(hpc.dg_ca3_weights.get_value())
        self.assertEqual(matrix_sum, no_of_columns*hpc.dg_ca3_weights.get_value().shape[0], msg="Sum not cols*rows.")

    def test_all_columns_sum_ca3_ca3(self):
        hpc = HPC([49, 240, 1600, 480, 49],
          0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
          0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
          0.7, 1, 0.1, 0.5,  # gamma, epsilon, nu, turnover rate
          0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha

        no_of_columns = hpc.ca3_ca3_weights.get_value().shape[1]

        empty_array = np.zeros(1, hpc.ca3_ca3_weights.shape[0]).astype(np.float32)
        for column_index in range(no_of_columns):
            hpc.update_ca3_ca3_weights_column(column_index, empty_array)

        matrix_sum = np.sum(hpc.ca3_ca3_weights.get_value())
        self.assertEqual(matrix_sum, 0, msg="Sum was not zero after all elements set to zero. Sum: "+str(matrix_sum))

        full_array = np.ones_like(empty_array).astype(np.float32)

        # set one column only:
        num_of_column_elements = hpc.ca3_ca3_weights.get_value().shape[0]
        hpc.update_ca3_ca3_weights_column(5, full_array)
        self.assertEqual(np.sum(hpc.ca3_ca3_weights.get_value()), num_of_column_elements,
                         msg="Matrix sum not equal to the number of column elements.")

        for column_index in range(no_of_columns):
            hpc.update_ca3_ca3_weights_column(column_index, full_array)

        matrix_sum = np.sum(hpc.ca3_ca3_weights.get_value())
        self.assertEqual(matrix_sum, no_of_columns*hpc.ca3_ca3_weights.get_value().shape[0], msg="Sum not cols*rows.")

    def test_all_columns_sum_ca3_out(self):
        hpc = HPC([49, 240, 1600, 480, 49],
          0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
          0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
          0.7, 1, 0.1, 0.5,  # gamma, epsilon, nu, turnover rate
          0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha

        no_of_columns = hpc.ca3_out_weights.get_value().shape[1]

        empty_array = np.zeros(1, hpc.ca3_out_weights.shape[0]).astype(np.float32)
        for column_index in range(no_of_columns):
            hpc.update_ca3_out_weights_column(column_index, empty_array)

        matrix_sum = np.sum(hpc.ca3_out_weights.get_value())
        self.assertEqual(matrix_sum, 0, msg="Sum was not zero after all elements set to zero. Sum: "+str(matrix_sum))

        full_array = np.ones_like(empty_array).astype(np.float32)

        # set one column only:
        num_of_column_elements = hpc.ca3_out_weights.get_value().shape[0]
        hpc.update_ca3_out_weights_column(5, full_array)
        self.assertEqual(np.sum(hpc.ca3_out_weights.get_value()), num_of_column_elements,
                         msg="Matrix sum not equal to the number of column elements.")

        for column_index in range(no_of_columns):
            hpc.update_ca3_out_weights_column(column_index, full_array)

        matrix_sum = np.sum(hpc.ca3_out_weights.get_value())
        self.assertEqual(matrix_sum, no_of_columns*hpc.ca3_out_weights.get_value().shape[0], msg="Sum not cols*rows.")

class TestWeightUpdateCa3Out(unittest.TestCase):

    def test_equation_ca3_out(self):
        hpc = HPC([49, 240, 1600, 480, 49],
                  0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
                  0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
                  0.7, 1, 0.1, 0.5,  # gamma, epsilon, nu, turnover rate
                  0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha

        empty_activation_values_l1 = np.zeros_like(hpc.ca3_values.get_value()).astype(np.float32)
        empty_activation_values_l1.put([0, 0], 1)
        hpc.set_ca3_values(empty_activation_values_l1)

        empty_activation_values_l2 = np.zeros_like(hpc.output_values.get_value()).astype(np.float32)
        empty_activation_values_l2.put([0, 0], 1)
        hpc.set_output(empty_activation_values_l2)

        current_weight_element = hpc.ca3_out_weights.get_value()[0][0]
        next_weight_element = hpc._gamma * current_weight_element + 1

        hpc.wire_ca3_out_wrapper()
        self.assertAlmostEqual(hpc.ca3_out_weights.get_value()[0][0], next_weight_element, places=6,
                         msg="Weight update did not correspond to the predicted update value of the equation: "
                             "next_weight_el != w_el : "+str(next_weight_element)+" != " +
                             str(hpc.ca3_out_weights.get_value()[0][0]))

    def test_equation_ca3_ca3(self):
        hpc = HPC([49, 240, 1600, 480, 49],
                  0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
                  0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
                  0.7, 1, 0.1, 0.5,  # gamma, epsilon, nu, turnover rate
                  0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha

        empty_activation_values_l1 = np.zeros_like(hpc.ca3_values.get_value()).astype(np.float32)
        empty_activation_values_l1.put([0, 0], 1)
        hpc.set_ca3_values(empty_activation_values_l1)

        current_weight_element = hpc.ca3_ca3_weights.get_value()[0][0]
        next_weight_element = hpc._gamma * current_weight_element + 1

        hpc.wire_ca3_ca3_wrapper()
        self.assertAlmostEqual(hpc.ca3_ca3_weights.get_value()[0][0], next_weight_element, places=6,
                         msg="Weight update did not correspond to the predicted update value of the equation: "
                             "next_weight_el != w_el : "+str(next_weight_element)+" != " +
                             str(hpc.ca3_ca3_weights.get_value()[0][0]))

    def test_equation_ec_ca3(self):
        hpc = HPC([49, 240, 1600, 480, 49],
                  0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
                  0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
                  0.7, 1, 0.1, 0.5,  # gamma, epsilon, nu, turnover rate
                  0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha

        empty_activation_values_l1 = np.zeros_like(hpc.ec_values.get_value()).astype(np.float32)
        empty_activation_values_l1.put([0, 0], 1)
        hpc.set_ca3_values(empty_activation_values_l1)

        empty_activation_values_l2 = np.zeros_like(hpc.ca3_values.get_value()).astype(np.float32)
        empty_activation_values_l2.put([0, 0], 1)
        hpc.set_ca3_values(empty_activation_values_l2)

        current_weight_element = hpc.ec_ca3_weights.get_value()[0][0]
        next_weight_element = current_weight_element + hpc._nu * (1 - current_weight_element)

        hpc.wire_ec_ca3_wrapper()
        self.assertAlmostEqual(hpc.ec_ca3_weights.get_value()[0][0], next_weight_element, places=6,
                         msg="Weight update did not correspond to the predicted update value of the equation: "
                             "next_weight_el != w_el : "+str(next_weight_element)+" != " +
                             str(hpc.ec_ca3_weights.get_value()[0][0]))





if __name__ == '__main__':
    unittest.main()
