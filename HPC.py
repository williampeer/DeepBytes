import theano
import theano.tensor as T
import numpy as np

# Note: Ensure float32 for GPU-usage. Use the profiler to analyse GPU-usage.
_GAMMA = 0.3

class HPC:
    def __init__(self):
        neuron_numbers = [5, 20, 20, 20, 5]

        # vectors for neuron activation values.
        self.input_values = T.fvector(neuron_numbers[0])
        self.ec_values = T.fvector(neuron_numbers[1])
        self.dg_values = T.fvector(neuron_numbers[2])
        self.ca3_values = T.fvector(neuron_numbers[3])
        self.output_values = T.fvector(neuron_numbers[4])

        # initialise weight matrices.
        self.input_ec_weights = T.fmatrix(neuron_numbers[0], neuron_numbers[1])
        self.ec_dg_weights = T.fmatrix(neuron_numbers[1], neuron_numbers[2])
        self.ec_ca3_weights = T.fmatrix(neuron_numbers[1], neuron_numbers[3])
        self.ca3_ca3_weights = T.fmatrix(neuron_numbers[3], neuron_numbers[3])
        self.ca3_output_weights = T.fmatrix(neuron_numbers[3], neuron_numbers[4])

        # setup Theano functions
        new_input = T.fvector
        self.set_input = theano.function(new_input, outputs=None,
                                         updates=(self.input_values, new_input))

        # Hebbian learning weight updates for input -> EC
        activation_prod_in_ec_matrix = T.transpose(self.input_values) * self.ec_values
        # check dims:
        # print T.dim(activation_prod_matrix) ?
        self.ec_layer_pass = theano.function([activation_prod_in_ec_matrix, self.input_ec_weights], outputs=None,
                                             updates=[(self.input_ec_weights, _GAMMA * self.input_ec_weights +
                                                       activation_prod_in_ec_matrix),
                                                      (self.ec_values, self.input_values * self.input_ec_weights)])


    def iter(self):
        # one iter for each part, such as:
        pass
