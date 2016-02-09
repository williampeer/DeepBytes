import theano
import theano.tensor as T
import numpy as np
from Tools import binomial_f, uniform_f
theano.config.floatX = 'float32'


# Ans et al. (1997): Error measure the cross-entropy. Learning rate: 0.01, momentum term: 0.5.
class SimpleNeocorticalNetwork:
    def __init__(self, in_dim, h_dim, out_dim, alpha, momentum):

        self.alpha = alpha
        self.momentum = momentum

        self.dims = [in_dim, h_dim, out_dim]

        _in = np.zeros((1, in_dim), dtype=np.float32)
        _h = np.zeros((1, h_dim), dtype=np.float32)
        _out = np.zeros((1, out_dim), dtype=np.float32)

        self._in = theano.shared(name='_in', value=_in.astype(theano.config.floatX), borrow=True)
        self._h = theano.shared(name='_h', value=_h.astype(theano.config.floatX), borrow=True)
        self._out = theano.shared(name='_out', value=_out.astype(theano.config.floatX), borrow=True)

        in_h_Ws = uniform_f(in_dim, h_dim)
        h_out_Ws = uniform_f(h_dim, out_dim)

        self.in_h_Ws = theano.shared(name='in_h_Ws', value=in_h_Ws.astype(theano.config.floatX), borrow=True)
        self.h_out_Ws = theano.shared(name='h_out_Ws', value=h_out_Ws.astype(theano.config.floatX), borrow=True)

        prev_dW1 = np.zeros_like(in_h_Ws, dtype=np.float32)
        prev_dW2 = np.zeros_like(h_out_Ws, dtype=np.float32)
        self.prev_delta_W_in_h = theano.shared(name='prev_delta_W_in_h', value=prev_dW1.astype(theano.config.floatX),
                                               borrow=True)
        self.prev_delta_W_h_out = theano.shared(name='prev_delta_W_h_out', value=prev_dW2.astype(theano.config.floatX),
                                                borrow=True)

        new_input = T.fmatrix()
        input_hidden_Ws = T.fmatrix()
        hidden_output_Ws = T.fmatrix()
        sum_h = new_input.dot(input_hidden_Ws)
        next_h = T.tanh(sum_h)
        sum_out = next_h.dot(hidden_output_Ws)
        next_out = T.tanh(sum_out)

        self.feed_forward = theano.function([new_input, input_hidden_Ws, hidden_output_Ws],
                                            updates=[(self._in, new_input), (self._h, next_h), (self._out, next_out)])

        Ws_h_out = T.fmatrix()
        Ws_in_h = T.fmatrix()
        prev_delta_W_in_h = T.fmatrix()
        prev_delta_W_h_out = T.fmatrix()
        o_in = T.fmatrix()
        o_h = T.fmatrix()
        o_out = T.fmatrix()
        target_out = T.fmatrix()

        # L2 norm
        tmp = o_out-target_out
        error = tmp

        tmp_grad_h_out = np.ones_like(o_out, dtype=np.float32) / T.cosh(o_out)
        diracs_out = error * tmp_grad_h_out * tmp_grad_h_out
        delta_W_h_out = - self.alpha * o_h.T.dot(diracs_out) + self.momentum * prev_delta_W_h_out
        new_Ws_h_out = Ws_h_out + delta_W_h_out

        tmp_grad_in_h = np.ones_like(o_h, dtype=np.float32) / T.cosh(o_h)
        diracs_h_layer_terms = tmp_grad_in_h * tmp_grad_in_h
        diracs_h_chain = diracs_out.dot(Ws_h_out.T)
        diracs_h = diracs_h_chain * diracs_h_layer_terms
        delta_W_in_h = - self.alpha * o_in.T.dot(diracs_h) + self.momentum * prev_delta_W_in_h
        new_Ws_in_h = Ws_in_h + delta_W_in_h

        self.back_propagate = theano.function([Ws_in_h, Ws_h_out, o_in, o_h, o_out, target_out,
                                               prev_delta_W_in_h, prev_delta_W_h_out],
                                              updates=[(self.h_out_Ws, new_Ws_h_out), (self.in_h_Ws, new_Ws_in_h),
                                                       (self.prev_delta_W_in_h, delta_W_in_h),
                                                       (self.prev_delta_W_h_out, delta_W_h_out)])

        # self.set_input = theano.function([new_input], updates=[(self._in, new_input)])

    def train(self, IOPairs):
        for input_pattern, output_pattern in IOPairs:

            # no learning criteria, only propagate once?
            self.feed_forward(input_pattern, self.in_h_Ws.get_value(return_internal_type=True),
                              self.h_out_Ws.get_value(return_internal_type=True))
            self.back_propagate(self.in_h_Ws.get_value(return_internal_type=True),
                                self.h_out_Ws.get_value(return_internal_type=True),
                                self._in.get_value(return_internal_type=True),
                                self._h.get_value(return_internal_type=True),
                                self._out.get_value(return_internal_type=True), output_pattern,
                                self.prev_delta_W_in_h.get_value(return_internal_type=True),
                                self.prev_delta_W_h_out.get_value(return_internal_type=True))

    def get_pseudopattern_I(self):
        # random input
        random_in_pattern = binomial_f(1, self.dims[0], 0.5)
        random_in_pattern = random_in_pattern * 2 - np.ones_like(random_in_pattern)
        # print "random_in_pattern:", random_in_pattern

        return self.get_IO(random_in_pattern)

    def get_IO(self, input_pattern):
        self.feed_forward(input_pattern, self.in_h_Ws.get_value(), self.h_out_Ws.get_value())

        corresponding_output = self._out.get_value()
        return [input_pattern, corresponding_output]

    def print_layers(self):
        print "\nPrinting layer activation values:"
        print "input:\t", self._in.get_value()
        print "hidden:\t", self._h.get_value()
        print "output:\t", self._out.get_value()