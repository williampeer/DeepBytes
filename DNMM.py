import theano
import theano.tensor as T
import time
from HPC import *
from SimpleNeocorticalNetwork import *

# ==================== TESTING CODE: ======================
# Hippocampal module
hpc = HPC([32, 240, 1600, 480, 32],
          0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
          0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
          0.7, 0.1, 1, 0.5,  # gamma, epsilon, nu, turnover rate
          0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha, alpha is 2, 4, and 5 in different experiments in Hattori (2014)
# sample IO:
c = np.asarray([[1, -1] * 16], dtype=np.float32)
d = np.asarray([[-1, 1] * 16], dtype=np.float32)
hpc.set_input(c)
hpc.set_output(d)
# hpc.print_info()
time_before = time.time()
for i in xrange(10):
     hpc.iter()
     # hpc.print_info()
time_after = time.time()
train_time = time_after - time_before

time_before = time.time()
hpc.iter_until_stopping_criteria()
time_after = time.time()
prop_time_until_stable = time_after - time_before
print "output:", hpc.output_values.get_value()

print "Training time: ", train_time
print "Propagation time until stability:", prop_time_until_stable



# Neocortical module:
# ann = SimpleNeocorticalNetwork(32, 50, 32, 0.85, 0.01)
#
# # a = np.random.random((1, 32)).astype(np.float32)
# # b = -1 * np.random.random((1, 32)).astype(np.float32)
# a = np.asarray([[0.1, 0.2] * 16], dtype=np.float32)
# b = np.asarray([[-0.2, -0.4] * 16], dtype=np.float32)
#
# iopair = [a, b]
#
# print "target output:", b
# for i in range(20000):
#     ann.train([iopair])
# print ann.in_h_Ws.get_value()
# print ann.h_out_Ws.get_value()
# ann.print_layers()