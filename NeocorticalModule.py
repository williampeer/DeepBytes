import theano
import theano.tensor as T
import numpy as np

theano.config.floatX = 'float32'


# This can be completed if we want a neocortical network of arbitrary size. For now, the simple version is optimized by
#   slightly more hard-coding.
class NeocorticalNetwork:
    def __init__(self, layers_dims,
                 learning_rate, momentum_term):
        self.num_of_layers = len(layers_dims)
        print "self.num_of_layers:", self.num_of_layers
        self.learning_rate = learning_rate
        self.momentum_term = momentum_term

        # ============================== SHARED VARIABLES ======================================
        self.layers = []
        for i in range(self.num_of_layers):
            layer = np.zeros((1, layers_dims[i]), dtype=np.float32)
            layer_shared = theano.shared(value=layer.astype(theano.config.floatX))
            self.layers.append(layer_shared)
        print "len(self.layers):", len(self.layers)

        self.Ws = []
        for next_layer_index in range(1, self.num_of_layers-1):
            Ws_layer_i_j = np.random.random((layers_dims[next_layer_index-1], layers_dims[next_layer_index])).astype(
                np.float32)
            Ws_layer_i_j_shared = theano.shared(value=Ws_layer_i_j.astype(theano.config.floatX))
            self.Ws.append(Ws_layer_i_j_shared)

        # ================================== FUNCTIONS ==========================================
        new_input = T.fmatrix('new_input')
        self.set_input = theano.function([new_input], updates=[(self.layers[0], new_input)])

        # Euclidean distance
        target_output = T.fmatrix('target_output')
        self.calculate_error = theano.function([target_output], outputs=np.power(self.layers[self.num_of_layers-1] -
                                                                                   target_output, 2)/2)

        # _, scan_updates = theano.scan(fn=)
        # scan_feed_forward = theano.function([], updates=[(self.layers, scan_updates)])

        m1 = T.fmatrix()
        m2 = T.fmatrix()
        self.dot_product = theano.function([m1, m2], outputs=m1.dot(m2))

    def feed_forward(self):
        print "FF"
        for index in range(self.num_of_layers-1):  # 0-indexed
            print "index:", index
            update_next_layer = theano.function([], updates=[(self.layers[index+1],
                self.layers[index].get_value(borrow=True).dot(self.Ws[index].get_value(borrow=True)))])
            update_next_layer()
            print "updated layer of index+1:", self.layers[index+1]

    def back_propagate(self, target_output):
        error_array = self.calculate_error(target_output)
        for weight_index in range(self.num_of_layers-2, 0, -1):
            delta_W = - self.learning_rate * T.grad(error_array, wrt=self.Ws[weight_index])
            self.Ws[weight_index] = self.Ws[weight_index] + delta_W

    def train(self, IOPairs):
        for pair in IOPairs:
            input_pattern = pair[0]
            output_pattern = pair[1]
            # no learning criteria, only propagate once?
            self.set_input(input_pattern)
            self.feed_forward()
            self.back_propagate(output_pattern)

    def print_layers(self):
        for layer in self.layers:
            print layer

neocorticalNetwork = NeocorticalNetwork([10, 11, 12], 0.01, 0.9)
neocorticalNetwork.set_input(np.asarray([range(10)], dtype=np.float32))
neocorticalNetwork.feed_forward()
