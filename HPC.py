import theano
import theano.tensor as T
import numpy as np

# Note: Ensure float32 for GPU-usage. Use the profiler to analyse GPU-usage.
_GAMMA = 0.3

class HPC:
    def __init__(self, dims):
        # in, ec, dg, ca3, out
        neuron_numbers = dims

        # vectors for neuron activation values.
        self.input_values = theano.shared(np.random.random((1, neuron_numbers[0])).astype(np.float32), 'input_values',
                                          True)
        self.ec_values = theano.shared(np.random.random((1, neuron_numbers[1])).astype(np.float32), 'ec_values', True)

        # print self.input_values.get_value()
        # print self.ec_values.get_value()
        #self.dg_values = T.fvector('dg_values')
        #self.ca3_values = T.fvector('ca3_values')
        #self.output_values = T.fvector('output_values')

        # initialise weight matrices.
        self.in_ec_weights = theano.shared(np.random.random((neuron_numbers[0], neuron_numbers[1])).astype(np.float32),
                                           name='in_ec_weights', borrow=True)

        # print self.in_ec_weights.get_value()
        #self.ec_dg_weights = T.fmatrix('ec_dg_weights')
        #self.dg_ca3_weights = T.fmatrix('dg_ca3_weights')
        #self.ec_ca3_weights = T.fmatrix('ec_ca3_weights')
        #self.ca3_ca3_weights = T.fmatrix('ca3_ca3_weights')
        #self.ec_out_weights = T.fmatrix('ca3_out_weights')

        # setup Theano functions
        new_input = T.fmatrix('new_input')
        self.set_input = theano.function([new_input], outputs=None,
                                         updates=[(self.input_values, new_input)])

        m1 = T.fmatrix('m1')
        m2 = T.fmatrix('m2')
        result = m1.dot(m2)
        dot_product = theano.function([m1, m2], outputs=result)

        # print dot_product(np.random.random((1, 5)).astype(np.float32), np.random.random((5, 20)).astype(np.float32))
        # print "dot product:", dot_product(self.input_values.get_value(), self.in_ec_weights.get_value())

        # Hebbian learning weight updates for input -> EC
        # activation_prod_in_ec_matrix = T.transpose(self.input_values) * self.ec_values
        # check dims:
        # print T.dim(activation_prod_matrix) ?
        self.in_ec_pass = theano.function([], outputs=None,
                                          updates=[(self.in_ec_weights, _GAMMA * self.in_ec_weights +
                                                    T.transpose(self.input_values).dot(self.ec_values)),
                                                   (self.ec_values, self.input_values.dot(self.in_ec_weights))])

        # # activation_prod_ec_dg = T.transpose(self.ec_values) * self.dg_values
        # self.ec_dg_pass = theano.function([], outputs=None,
        #                                   updates=[(self.ec_dg_weights, _GAMMA * self.ec_dg_weights +
        #                                             T.transpose(self.ec_values) * self.dg_values),
        #                                            (self.dg_values, self.ec_values * self.ec_dg_weights)])
        #
        # # TODO: This needs to be consistent. Rewrite to one update function to ensure consistency?
        # #   (After applying the constrained Hebbian learning function)
        # # activation_prod_dg_ca3 = T.transpose(self.dg_values) * self.ca3_values
        # self.dg_ca3_pass = theano.function([], outputs=None,
        #                                    updates=[(self.dg_ca3_weights, _GAMMA * self.dg_ca3_weights +
        #                                              T.transpose(self.dg_values) * self.ca3_values),
        #                                             (self.ca3_values, self.dg_values * self.dg_ca3_weights)])
        # # act_prod_ca3_ca3 = T.transpose(self.ca3_values) * self.ca3_values
        # self.ca3_ca3_pass = theano.function([], outputs=None,
        #                                     updates=[(self.ca3_ca3_weights, _GAMMA * self.ca3_ca3_weights +
        #                                               T.transpose(self.ca3_values) * self.ca3_values),
        #                                              (self.ca3_values, self.ca3_values * self.ca3_ca3_weights)])
        #
        # # act_prod_ca3_out = T.transpose(self.ca3_values) * self.output_values
        # self.ca3_out_pass = theano.function([], outputs=None,
        #                                     updates=[(self.ec_out_weights, _GAMMA * self.ec_out_weights +
        #                                               T.transpose(self.ca3_values) * self.output_values),
        #                                              (self.output_values, self.ca3_values * self.ec_out_weights)])


    # TODO: Check parallelism. Check further decentralization possibilities.
    def iter(self):
        # one iter for each part, such as:
        self.in_ec_pass()
        #self.ec_dg_pass()
        #self.dg_ca3_pass()
        #self.ca3_ca3_pass()
        #self.ca3_out_pass()

    def print_info(self):
        print "\nprinting in and ec values:"
        print hpc.input_values.get_value(), "\n", hpc.ec_values.get_value()
        print "\n"
        print "weights:\n", hpc.in_ec_weights.get_value()


# testing code:

hpc = HPC([5, 20, 20, 20, 5])
hpc.print_info()
hpc.iter()
hpc.print_info()
