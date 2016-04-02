# from HPC import *
# from Tools import save_images_from
# import cPickle
# import time
import sys
#
# io_dim = 49
#
# hpc = HPC([io_dim, 240, 1600, 480, io_dim],
#           0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
#           0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
#           0.7, 100, 0.1, 0.5,  # gamma, epsilon, nu, turnover rate
#           0.10, 0.95, 0.8, 2.0, 3)  # k_m, k_r, a_i, alpha
#
# I = np.asarray([[1, -1, 1, -1, 1, -1, 1] * 7], dtype=np.float32)
#
# t1 = time.time()
# ctr, found_stable_output, out_now = hpc.recall_until_stability_criteria(False, 300)
# t2 = time.time()
#
# print "Time:", str(t2-t1)

test = sys.stdin.readline()
print "test:", test

# test_vals = np.random.random((1, 100)).astype(np.float32)
# test_vals = 0.312987 * np.ones((1, 100), dtype=np.float32)
# t0 = time.time()
# result = kWTA(test_vals, 0.20)
# t1 = time.time()
# print result
# print "np.sum(result[0]):", np.sum(result[0])
# print "kWTA in", "{:7.3f}".format(t1-t0), "seconds."

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
