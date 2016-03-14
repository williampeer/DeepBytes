from HPC import *
from Tools import save_images_from
import Tools
import cPickle

io_dim = 49
theano.config.floatX = 'float32'

hpc = HPC([io_dim, 240, 1600, 480, io_dim],
          0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
          0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
          0.7, 1, 0.1, 0.5,  # gamma, epsilon, nu, turnover rate
          0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha

# I = np.asarray([[1, -1, 1, -1, 1, -1, 1] * 7], dtype=np.float32)
# save_images_from([I])

l1 = theano.shared(value=Tools.binomial_f(1, 49, 0.5).astype('float32'), borrow=True)
l2 = theano.shared(value=np.zeros_like(l1, dtype=np.float32).astype('float32'), borrow=True)
n_dim = l1.get_value().shape[1]
print("n_dim:", n_dim)  # ->49
Ws = theano.shared(value=Tools.binomial_f(n_dim, n_dim, 0.5).astype('float32'), borrow=True)
l2_firing_rate = 0.1023

l1_in = T.fmatrix()
Ws_in = T.fmatrix()
next_l2_vals = l1_in.dot(Ws_in)
fire_to_l2 = theano.function([l1_in, Ws_in], updates=[(l2, next_l2_vals)])

# fire_to_l2(l1.get_value(), Ws.get_value())
print("l2 vals before kWTA:", l2.get_value())
# l2 = hpc.kWTA(l2.get_value(), l2_firing_rate)
# print("l2:", l2.get_value())


# test_vals = np.random.random((1, 100)).astype(np.float32)
# test_vals = 0.312987 * np.ones((1, 100), dtype=np.float32)
# t0 = time.time()
# result = hpc.kWTA(test_vals, 0.20)
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
