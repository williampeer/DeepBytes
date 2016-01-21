import theano
import theano.tensor as T
import numpy as np

theano.config.floatX = 'float32'

class NeocorticalNetwork():
    def __init__(self, layers_dims,
                 learning_rate, momentum_term):
        self.num_of_layers = len(layers_dims)
        self.learning_rate = learning_rate
        self.momentum_term = momentum_term

        # ============================== SHARED VARIABLES ======================================
        self.layers = []
        for i in range(self.num_of_layers):
            layer = np.zeros((1, layers_dims[i]), dtype=np.float32)
            layer_shared = theano.shared(value=layer.astype(theano.config.floatX))
            self.layers.append(layer_shared)

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
        calculate_error = theano.function([target_output], outputs=T.sum(np.power(self.layers[self.num_of_layers-1] -
                                                                                   target_output, 2)))

        W_i = T.fmatrix()
        l_i = T.fmatrix()
        propagate_to_next_layer = theano.function([l_i, W_i], outputs=l_i.dot(W_i))

        results_ff, updates_ff = theano.scan(fn=propagate_to_next_layer, non_sequences=[self.layers, self.Ws],
                                             n_steps=self.num_of_layers-1)

    def backprop(self, target_output):
        # error = self.error_measure(self.output_layer, target_output)
        pass

    def train(self, IOPairs):
        for pair in IOPairs:
            input_pattern = pair[0]
            output_pattern = pair[1]
            # no learning criteria, only propagate once?
            self.set_input(input_pattern)
            self.feed_forward()
            self.backprop(output_pattern)

    def print_layers(self):
        for layer in self.layers:
            print layer

neocorticalNetwork = NeocorticalNetwork([10, 11, 12], 0.01, 0.9)
neocorticalNetwork.set_input(np.asarray([range(10)], dtype=np.float32))
print neocorticalNetwork.layers[0].get_value()
neocorticalNetwork.print_layers()
neocorticalNetwork.feed_forward()
neocorticalNetwork.print_layers()