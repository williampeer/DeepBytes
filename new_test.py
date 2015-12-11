import theano
import theano.tensor as T
import numpy as np


def floatX(x):
        return np.asarray(x,dtype=theano.config.floatX)


def init_matrix(shape):
        return floatX(np.random.randn(*shape))

x = T.matrix()
y = x ** 4
f = theano.function([x], y)

#print f(init_matrix([3, 3]))
print f(np.random.rand(3, 3))

theano.printing.pydotprint(y, outfile="/symbolic_graph_unopt.png",
                           var_with_name_simple=True)
theano.printing.pydotprint(f, outfile="/symbolic_graph_opt.png",
                           var_with_name_simple=True)