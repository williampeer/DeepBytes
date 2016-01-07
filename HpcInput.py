import theano
import theano.Tensor as T
import numpy as np

class HpcInput:
    def __init__(self, input_size, ec_size):
        self.input_array = T.fvector(input_size)
        self.weights = T.fmatrix(input_size, ec_size)
