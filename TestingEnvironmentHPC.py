from HPC import *
from data_capital import *
import time

io_dim = 49

hpc = HPC([io_dim, 240, 1600, 480, io_dim],
          0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
          0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
          0.7, 1, 0.1, 0.5,  # gamma, epsilon, nu, turnover rate
          0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha


def kWTA(values, f_r):
    # print "values[0]", values[0]
    values_length = len(values[0])
    k = np.round(values_length * f_r).astype(np.int32)

    sort_values_f = theano.function([], outputs=T.sort(values))
    sorted_values = sort_values_f()
    k_th_largest_value = sorted_values[0][values_length-k-1]
    print "sorted_values:", sorted_values
    print "k_th_largest_value:", k_th_largest_value

    mask_vector = k_th_largest_value * np.ones_like(values)
    result = (values >= mask_vector).astype(np.float32)
    print result

    return result


test_vals = np.random.random((1, 100)).astype(np.float32)
t0 = time.time()
result = kWTA(test_vals, 0.20)
# print result
print "np.sum(result[0]):", np.sum(result[0])
t1 = time.time()
print "kWTA in", "{:7.3f}".format(t1-t0), "seconds."

# t0 = time.time()
# hpc.neuronal_turnover_dg()
# t1 = time.time()
# print "nt in:", "{:7.3f}".format(t1-t0), "seconds."
#
# I = np.asarray([[1, -1, 1, -1, 1, -1, 1] * 7], dtype=np.float32)
# O = np.asarray([[-1, 1, -1, 1, -1, 1, -1] * 7], dtype=np.float32)
# patterns = []
# for pattern in data_letters_capital[:4]:
#     io = [[]]
#     for row in pattern:
#         for el in row:
#             io[0].append(el)
#     new_array = np.asarray(io, dtype=np.float32)
#     patterns.append([new_array, new_array])
#
# print "\nsetup"
# hpc.setup_pattern(patterns[0][0], patterns[0][0])
# print "init. output:", hpc.output_values.get_value()
# hpc.print_activation_values_sum()
#
# print "\nlearn"
# for i in range(1):
#     hpc.learn()
# print "output after learn():", hpc.output_values.get_value()
# hpc.print_activation_values_sum()
#
# print "\nrecall"
# hpc.recall()
# print "output after recall():", hpc.output_values.get_value()
# hpc.print_activation_values_sum()
