import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
from HPC import *

shared_random_generator = RandomStreams()
in_m = T.fmatrix()
binomial_vector_f = theano.function([in_m], outputs=shared_random_generator.binomial(size=in_m.shape, n=1, p=0.2,
                                                                                     dtype='float32'))

x = T.iscalar()
y = T.iscalar()
uniform_f = theano.function([x,y], outputs=shared_random_generator.uniform(size=(x,y), low=-1, high=1,
                                                                              dtype='float32'))

# m = np.arange(7).astype(np.float32)
# print binomial_vector_f([m])
# print uniform_f(2, 10)

hpc = HPC([49, 240, 1600, 480, 49],
          0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
          0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
          0.7, 1.0, 0.1, 0.5,  # gamma, epsilon, nu, turnover rate
          0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha, alpha is 2 in 4.1.1
