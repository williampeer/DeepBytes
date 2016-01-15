import theano
import theano.tensor as T
import numpy as np

# Note: Ensure float32 for GPU-usage. Use the profiler to analyse GPU-usage.
_GAMMA = 0.3
_EPSILON = 0.8
_LAMBDA = 0.4

class HPC:
    def __init__(self, dims):
        # in, ec, dg, ca3, out
        self.init_layers(dims)

        # setup Theano functions
        new_input = T.fmatrix('new_input')
        self.set_input = theano.function([new_input], outputs=None,
                                         updates=[(self.input_values, new_input)])
        new_output = T.fmatrix('new_output')
        self.set_output = theano.function([new_output], outputs=None,
                                         updates=[(self.output_values, new_output)])
        m1 = T.fmatrix('m1')
        m2 = T.fmatrix('m2')
        result = m1.dot(m2)
        self.dot_product = theano.function([m1, m2], outputs=result)

        theta = T.fscalar('theta')
        f_theta = T.tanh(theta/_EPSILON)
        self.transfer_function = theano.function([theta], outputs=f_theta)

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

    def init_layers(self, dims):
        # ============== ACTIVATION VALUES ==================
        # self.input_values = theano.shared(np.asarray([[1, 1]]).astype(np.float32), 'input_values', True)
        # self.ec_values = theano.shared(np.asarray([[0.5, 0.5]]).astype(np.float32), 'ec_values', True)
        # self.in_ec_weights = theano.shared(np.asarray([[1, 1], [1, 1]]).astype(np.float32),
        #                                    name='in_ec_weights', borrow=True)
        self.input_values = theano.shared(np.random.random((1, dims[0])).astype(np.float32), 'input_values', True)
        self.ec_values = theano.shared(np.random.random((1, dims[1])).astype(np.float32), 'ec_values', True)
        self.dg_values = theano.shared(np.random.random((1, dims[2])).astype(np.float32), 'ec_values', True)
        self.ca3_values = theano.shared(np.random.random((1, dims[3])).astype(np.float32), 'ec_values', True)
        self.output_values = theano.shared(np.random.random((1, dims[4])).astype(np.float32), 'output_values', True)

        # ============== WEIGHT MATRICES ===================
        self.in_ec_weights = theano.shared(np.random.random((dims[0], dims[1])).astype(np.float32),
                                           name='in_ec_weights', borrow=True)
        self.ec_dg_weights = theano.shared(np.random.random((dims[1], dims[2])).astype(np.float32),
                                           name='ec_dg_weights', borrow=True)
        self.ec_ca3_weights = theano.shared(np.random.random((dims[1], dims[3])).astype(np.float32),
                                           name='ec_ca3_weights', borrow=True)
        self.dg_ca3_weights = theano.shared(np.random.random((dims[1], dims[3])).astype(np.float32),
                                           name='dg_ca3_weights', borrow=True)
        self.ca3_ca3_weights = theano.shared(np.random.random((dims[3], dims[3])).astype(np.float32),
                                           name='ca3_ca3_weights', borrow=True)
        self.ca3_out_weights = theano.shared(np.random.random((dims[3], dims[4])).astype(np.float32),
                                           name='ca3_out_weights', borrow=True)

        # ============== HEBBIAN LEARNING ==================
        # unconstrained:
        next_in_ec_weights = _GAMMA * self.in_ec_weights + T.transpose(self.input_values).dot(self.ec_values)
        self.in_ec_pass = theano.function([], outputs=None,
                                          updates=[(self.in_ec_weights, next_in_ec_weights),
                                                   (self.ec_values, self.input_values.dot(next_in_ec_weights))])

        next_ca3_out_weights = _GAMMA * self.ca3_out_weights + T.transpose(self.ca3_values).dot(self.output_values)
        # without output activation value updates during learning?
        self.ca3_out_pass = theano.function([], outputs=None,
                                          updates=[(self.ca3_out_weights, next_ca3_out_weights)])
        # should the output values be updated? : (self.output_values, self.input_values.dot(next_in_ec_weights))])

        # constrained:
        next_ec_dg_weights = self.ec_dg_weights + _LAMBDA * T.transpose(self.dg_values).dot(self.ec_values -
                                                                              self.dg_values.dot(self.ec_dg_weights))
        self.ec_dg_pass = theano.function([], outputs=None,
                                          updates=[(self.ec_dg_weights, next_ec_dg_weights),
                                                   (self.dg_values, self.ec_values.dot(next_ec_dg_weights))])

        next_ec_ca3_weights = self.ec_ca3_weights + _LAMBDA * T.transpose(self.ca3_values).dot(self.ec_values -
                                                                            self.ca3_values.dot(self.ec_ca3_weights))
        self.ec_ca3_pass = theano.function([], outputs=None,
                                          updates=[(self.ec_ca3_weights, next_ec_ca3_weights),
                                                   (self.ca3_values, self.ec_values.dot(next_ec_ca3_weights))])

        next_dg_ca3_weights = self.dg_ca3_weights + _LAMBDA * T.transpose(self.ca3_values).dot(self.dg_values -
                                                                            self.ca3_values.dot(self.dg_ca3_weights))
        self.dg_ca3_pass = theano.function([], outputs=None,
                                          updates=[(self.dg_ca3_weights, next_dg_ca3_weights),
                                                   (self.ca3_values, self.ec_values.dot(next_dg_ca3_weights))])

        next_ca3_ca3_weights = self.ca3_ca3_weights + _LAMBDA * T.transpose(self.ca3_values).dot(self.ca3_values -
                                                                            self.ca3_values.dot(self.ca3_ca3_weights))
        self.ca3_ca3_pass = theano.function([], outputs=None,
                                          updates=[(self.ca3_ca3_weights, next_ca3_ca3_weights),
                                                   (self.ca3_values, self.ca3_values.dot(next_ca3_ca3_weights))])


    # TODO: Check parallelism. Check further decentralization possibilities.
    def iter(self):
        # one iter for each part, such as:
        self.in_ec_pass()
        self.ec_dg_pass()
        self.dg_ca3_pass()
        self.ca3_ca3_pass()
        self.ca3_out_pass()

    def print_info(self):
        print "\nprinting activation values:"
        print hpc.input_values.get_value()
        print hpc.ec_values.get_value()
        print hpc.dg_values.get_value()
        print hpc.ca3_values.get_value()
        print hpc.output_values.get_value()

        print "\nweights:"
        print hpc.in_ec_weights.get_value()
        print hpc.ec_dg_weights.get_value()
        print hpc.ec_ca3_weights.get_value()
        print hpc.ca3_ca3_weights.get_value()
        print hpc.ca3_out_weights.get_value()


# testing code:

hpc = HPC([2, 20, 20, 20, 5])
# hpc.in_ec_pass()
# hpc.ec_dg_pass()
hpc.print_info()
hpc.iter()
hpc.print_info()
