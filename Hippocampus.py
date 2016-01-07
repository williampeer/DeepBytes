import theano
import theano.Tensor as T
import numpy as np

class HPC:
    def __init__(self, layer_dims):
        self.input = HpcInput()
        self.ec = EC()
        self.dg = DG()
        self.ca3 = CA3()
        self.output = HpcOutput()

    def iter(self):
        # one iter for each part, such as:
        self.input.iter()

