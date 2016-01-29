import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
from HPC import *

# shared_random_generator = RandomStreams()
# in_m = T.fmatrix()
# binomial_vector_f = theano.function([in_m], outputs=shared_random_generator.binomial(size=in_m.shape, n=1, p=0.2,
#                                                                                      dtype='float32'))
#
# x = T.iscalar()
# y = T.iscalar()
# uniform_f = theano.function([x,y], outputs=shared_random_generator.uniform(size=(x,y), low=-1, high=1,
#                                                                               dtype='float32'))
#
# # m = np.arange(7).astype(np.float32)
# # print binomial_vector_f([m])
# # print uniform_f(2, 10)
#
hpc = HPC([49, 240, 1600, 480, 49],
          0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
          0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
          0.7, 1.0, 0.1, 0.5,  # gamma, epsilon, nu, turnover rate
          0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha, alpha is 2 in 4.1.1
hpc.neuronal_turnover_dg_optimized()

# test = np.asarray([np.arange(5), np.arange(5)*2])
# shared_test = theano.shared(name='shared_test', value=test.astype(theano.config.floatX), borrow=True)
#
# index = T.iscalar()
# replacement = T.fvector()
# # updates_row = T.set_subtensor(shared_test[index, :], replacement)
# # updates_column = T.set_subtensor(test_column, replacement.T)
#
# replace_row = theano.function([index, replacement],
#                               updates={shared_test: T.set_subtensor(shared_test[index, :], replacement)})
# replace_column = theano.function([index, replacement],
#                                  updates={shared_test: T.set_subtensor(shared_test[:, index], replacement)})
#
# print "before update:\n", shared_test.get_value()
# replace_row(1, np.asarray([-1,-2,-3,-4,-5], dtype=np.float32))
# print "after row replacement\n:", shared_test.get_value()
# replace_column(2, np.asarray([42, 42], dtype=np.float32))
# print "after column update:\n", shared_test.get_value()
