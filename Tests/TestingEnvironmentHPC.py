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

rand_vals = Tools.binomial_f(1, 49, 0.5)
l1 = theano.shared(value=rand_vals.astype('float32'), borrow=True)
l2 = theano.shared(value=np.zeros_like(rand_vals, dtype=np.float32).astype('float32'), borrow=True)
n_dim = l1.get_value().shape[1]
print("n_dim:", n_dim)  # ->49
Ws = theano.shared(value=(Tools.uniform_f(n_dim, n_dim) * Tools.binomial_f(n_dim, n_dim, p_scalar=0.5)).  # 50 % conn.
                   astype('float32'), borrow=True)
l2_firing_rate = 0.1023

l1_in = T.fmatrix()
Ws_in = T.fmatrix()
next_l2_vals = l1_in.dot(Ws_in)
fire_to_l2 = theano.function([l1_in, Ws_in], updates=[(l2, next_l2_vals)])

fire_to_l2(l1.get_value(), Ws.get_value())
print("l2 vals before kWTA:", l2.get_value())

new_values = T.fmatrix()
l2_kWTA = theano.function([new_values], updates=[(l2, new_values)])
l2_kWTA(hpc.kWTA(l2.get_value(), l2_firing_rate))
print("l2:", l2.get_value())
print "l2 sum:", np.sum(l2.get_value())

column_index = T.iscalar("column_index")
new_Ws_column = T.fvector()
update_Ws_col = theano.function([new_Ws_column, column_index], updates={Ws: T.set_subtensor(Ws[:, column_index], new_Ws_column)})


def equation(index):
    _gamma = 0.9
    res = _gamma * Ws.get_value()[:, index] + l2.get_value()[0][index] * l1.get_value()
    return res[0]

# realistic scenario for after kWTA has been performed:
# weight updates for the winners only:
l2_vals = l2.get_value()
# print "l2_vals:", l2_vals
# print "equation(0):", equation(0)
# print "Ws before update:", Ws.get_value()
for val_index in range(l2_vals.shape[1]):
    if l2_vals[0][val_index]==1:
        # weights update: update column i
        update_Ws_col(equation(val_index), column_index=val_index)

# print "Ws after update:", Ws.get_value()

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
