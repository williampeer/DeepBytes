import theano
import theano.tensor as T
import numpy as np

# Note: Ensure float32 for GPU-usage. Use the profiler to analyse GPU-usage.
theano.config.floatX = 'float32'
_GAMMA = 0.3
_EPSILON = 1
_LAMBDA = 0.05

class HPC:
    def __init__(self, dims):
        # ============= setup Theano functions ==============
        m1 = T.fmatrix('m1')
        m2 = T.fmatrix('m2')
        result = m1.dot(m2)
        self.dot_product = theano.function([m1, m2], outputs=result)

        theta_ec = T.fmatrix('theta')
        f_theta = T.tanh(theta_ec/_EPSILON)
        self.transfer_function = theano.function([theta_ec], outputs=f_theta)

        # ============== ACTIVATION VALUES ==================
        input_values = np.random.uniform(-1, 1, (1, dims[0])).astype(np.float32)
        self.input_values = theano.shared(name='input_values', value=input_values.astype(theano.config.floatX),
                                          borrow=True)

        ec_values = np.random.uniform(0, 1, (1, dims[1])).astype(np.float32)
        self.ec_values = theano.shared(name='ec_values', value=ec_values.astype(theano.config.floatX), borrow=True)

        dg_values = np.random.uniform(0, 1, (1, dims[2])).astype(np.float32)
        self.dg_values = theano.shared(name='dg_values', value=dg_values.astype(theano.config.floatX), borrow=True)

        ca3_values = np.random.uniform(0, 1, (1, dims[3])).astype(np.float32)
        self.ca3_values = theano.shared(name='ca3_values', value=ca3_values.astype(theano.config.floatX), borrow=True)

        output_values = np.random.uniform(-1, 1, (1, dims[4])).astype(np.float32)
        self.output_values = theano.shared(name='output_values', value=output_values.astype(theano.config.floatX),
                                           borrow=True)

        # ============== WEIGHT MATRICES ===================
        input_ec_weights = np.ones((dims[0], dims[1])).astype(np.float32)
        # for each row, dims[0]: make 67 % of dims[1] 1, the rest 0:
        for row in range(dims[0]):
            for column in range(dims[1]):
                if np.random.random() < 0.33:
                    input_ec_weights[row][column] = 0
        self.in_ec_weights = theano.shared(name='in_ec_weights', value=input_ec_weights.astype(theano.config.floatX),
                                           borrow=True)

        ec_dg_weights = np.zeros((dims[1], dims[2])).astype(np.float32)
        # randomly assign about 25 % of the weights to a random connection weight
        for row in range(dims[1]):
            for col in range(dims[2]):
                if np.random.random() < 0.25:
                    ec_dg_weights[row][col] = np.random.random()
        self.ec_dg_weights = theano.shared(name='ec_dg_weights', value=ec_dg_weights.astype(theano.config.floatX),
                                           borrow=True)

        # randomly assign all weights between the EC and CA3
        ec_ca3_weights = np.random.random((dims[1], dims[3])).astype(np.float32)
        self.ec_ca3_weights = theano.shared(name='ec_ca3_weights', value=ec_ca3_weights.astype(theano.config.floatX),
                                            borrow=True)

        dg_ca3_weights = np.zeros((dims[2], dims[3])).astype(np.float32)
        # randomly assign about 4 % of the weights to random connection weights
        for row in range(dims[2]):
            for col in range(dims[3]):
                if np.random.random() < 0.04:
                    dg_ca3_weights[row][col] = np.random.random()
        self.dg_ca3_weights = theano.shared(name='dg_ca3_weights', value=dg_ca3_weights.astype(theano.config.floatX),
                                            borrow=True)

        # randomly assign 100 % of the weights
        ca3_ca3_weights = np.random.random((dims[3], dims[3])).astype(np.float32)
        self.ca3_ca3_weights = theano.shared(name='ca3_ca3_weights', value=ca3_ca3_weights.astype(theano.config.floatX),
                                             borrow=True)

        ca3_output_weights = np.random.random((dims[3], dims[4])).astype(np.float32)
        self.ca3_out_weights = theano.shared(name='ca3_out_weights',
                                             value=ca3_output_weights.astype(theano.config.floatX), borrow=True)

        # ============== HEBBIAN LEARNING ==================
        # Input:
        # Note: Use only one of the fire or fire_and_wire functions.
        # Apparently, weights are 0 or 1 between the input and EC and constant after initialization
        next_activation_values_ec = T.tanh(self.input_values.dot(self.in_ec_weights)/_EPSILON)
        self.propagate_input_to_ec = theano.function([], outputs=None, updates=[(self.ec_values,
                                                                                 next_activation_values_ec)])

        # ================= CONSTRAINED ====================
        next_activation_values_dg = T.tanh(self.ec_values.dot(self.ec_dg_weights)/_EPSILON)

        next_ec_dg_weights = self.ec_dg_weights + _LAMBDA * \
            (self.ec_values - (self.dg_values.dot(T.transpose(self.ec_dg_weights)))).T.dot(
                    next_activation_values_dg)

        self.fire_ec_dg = theano.function([], outputs=None, updates=[(self.dg_values, next_activation_values_dg)])
        self.fire_and_wire_ec_dg = theano.function([], outputs=None, updates=[(self.dg_values,
                                                                               next_activation_values_dg),
                                                                              (self.ec_dg_weights,
                                                                               next_ec_dg_weights)])

        theta_ec = self.ec_values.dot(self.ec_ca3_weights) + self.dg_values.dot(self.dg_ca3_weights) + \
            self.ca3_values.dot(self.ca3_ca3_weights)
        next_activation_values_ca3 = T.tanh(theta_ec/_EPSILON)
        self.fire_to_ca3 = theano.function([], outputs=None, updates=[(self.ca3_values, next_activation_values_ca3)])

        next_ec_ca3_weights = self.ec_ca3_weights + _LAMBDA * \
            (self.ec_values - next_activation_values_ca3.dot(T.transpose(self.ec_ca3_weights))).T.dot(
                    next_activation_values_ca3)

        next_dg_ca3_weights = self.dg_ca3_weights + _LAMBDA *\
            (self.dg_values - next_activation_values_ca3.dot(T.transpose(self.dg_ca3_weights))).T.dot(
                    next_activation_values_ca3)

        next_ca3_ca3_weights = _GAMMA * self.ca3_ca3_weights + \
            T.transpose(next_activation_values_ca3).dot(self.ca3_values)

        self.fire_and_wire_ca3 = theano.function([], outputs=None,
                                                 updates=[(self.ca3_values, next_activation_values_ca3),
                                                          (self.ec_ca3_weights, next_ec_ca3_weights),
                                                          (self.dg_ca3_weights, next_dg_ca3_weights),
                                                          (self.ca3_ca3_weights, next_ca3_ca3_weights)])
        # ===================================================
        # Output:
        next_activation_values_out = T.tanh(self.ca3_values.dot(self.ca3_out_weights)/_EPSILON)
        self.fire_ca3_out = theano.function([], outputs=None, updates=[(self.output_values,
                                                                        next_activation_values_out)])
        next_ca3_out_weights = _GAMMA * self.ca3_out_weights + T.transpose(self.ca3_values).\
            dot(self.output_values.get_value())
        self.wire_ca3_out = theano.function([], outputs=None, updates=[(self.ca3_out_weights, next_ca3_out_weights)])

        # =================== SETTERS =======================
        new_input = T.fmatrix('new_input')
        self.set_input = theano.function([new_input], outputs=None,
                                         updates=[(self.input_values, new_input)])
        new_output = T.fmatrix('new_output')
        self.set_output = theano.function([new_output], outputs=None, updates=[(self.output_values, new_output)])

    # TODO: Check parallelism. Check further decentralization possibilities.
    def iter(self):
        # one iter for each part, such as:
        self.propagate_input_to_ec()
        self.fire_and_wire_ec_dg()
        self.fire_and_wire_ca3()
        self.wire_ca3_out()

    def print_info(self):
        print "\nprinting activation values:"
        print self.input_values.get_value()
        print self.ec_values.get_value()
        print self.dg_values.get_value()
        print self.ca3_values.get_value()
        print self.output_values.get_value()

        print "\nweights:"
        print self.in_ec_weights.get_value()
        print self.ec_dg_weights.get_value()
        print self.ec_ca3_weights.get_value()
        print self.ca3_ca3_weights.get_value()
        print self.ca3_out_weights.get_value()


# testing code:

hpc = HPC([32, 240, 1600, 480, 32])
hpc.iter()
# sample IO:
# hpc.set_input(np.asarray([[1, 0, -1]]).astype(np.float32))
# hpc.set_output(np.asarray([[1, 0, -1]]).astype(np.float32))
# hpc.print_info()
# for i in xrange(1):
#     hpc.iter()