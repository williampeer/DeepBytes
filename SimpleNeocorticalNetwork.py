import theano
import theano.tensor as T
import numpy as np

theano.config.floatX = 'float32'


class SimpleNeocorticalNetwork:
    def __init__(self, in_dim, h_dim, out_dim):

        _in = np.random.random((1, in_dim)).astype(np.float32)
        _h = np.zeros((1, h_dim), dtype=np.float32)
        _out = np.zeros((1, out_dim), dtype=np.float32)

        self._in = theano.shared(name='_in', value=_in.astype(theano.config.floatX))
        self._h = theano.shared(name='_h', value=_h.astype(theano.config.floatX))
        self._out = theano.shared(name='_out', value=_out.astype(theano.config.floatX))

        in_h_Ws = np.random.random((in_dim, h_dim)).astype(np.float32)
        h_out_Ws = np.random.random((h_dim, out_dim)).astype(np.float32)

        self.in_h_Ws = theano.shared(name='in_h_Ws', value=in_h_Ws.astype(theano.config.floatX))
        self.h_out_Ws = theano.shared(name='h_out_Ws', value=h_out_Ws.astype(theano.config.floatX))

        new_input = T.fmatrix('new_input')
        self.set_input = theano.function([new_input], updates=[(self._in, new_input)])

        next_h = self._in.get_value(borrow=True, return_internal_type=True).dot(
                self.in_h_Ws.get_value(borrow=True, return_internal_type=True))
        next_out = next_h.dot(self.h_out_Ws.get_value(borrow=True, return_internal_type=True))

        self.feed_forward = theano.function([], updates=[(self._h, next_h), (self._out, next_out)])

        # L2 norm
        target_output = T.fmatrix('target_output')
        self.calculate_error = theano.function([target_output],
            outputs=np.power(self._out.get_value(borrow=True, return_internal_type=True) - target_output, 2)/2)

    def back_propagate(self, output_pattern):
        error_vector = self.calculate_error(output_pattern)
        print "err:", error_vector

    def train(self, IOPairs):
        for pair in IOPairs:
            input_pattern = pair[0]
            output_pattern = pair[1]

            # no learning criteria, only propagate once?
            self.set_input(input_pattern)
            self.feed_forward()
            # self.back_propagate(output_pattern)

    def print_layers(self):
        print "\nPrinting layer activation values:"
        print "input:\t", self._in.get_value()
        print "hidden:\t", self._h.get_value()
        print "output:\t", self._out.get_value()

ann = SimpleNeocorticalNetwork(3, 5, 3)

a = np.random.random((1, 3)).astype(np.float32)
b = np.random.random((1, 3)).astype(np.float32)

iopair = [a, b]

ann.print_layers()
ann.train([iopair])
ann.print_layers()
ann.back_propagate(b)