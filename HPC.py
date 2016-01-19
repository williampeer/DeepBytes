import theano
import theano.tensor as T
import numpy as np

# Note: Ensure float32 for GPU-usage. Use the profiler to analyse GPU-usage.
theano.config.floatX = 'float32'
_GAMMA = 0.3
_EPSILON = 1
_LAMBDA = 0.05
_TURNOVER_RATE = 0.50

class HPC:
    def __init__(self, dims, connection_rate_input_ec, perforant_path, mossy_fibers,
                 firing_rate_ec, firing_rate_dg, firing_rate_ca3):
        self.dims = dims  # numbers of neurons in the different layers
        self.connection_rate_input_ec = connection_rate_input_ec
        self.PP = perforant_path  # connection_rate_ec_dg
        self.MF = mossy_fibers  # connection_rate_dg_ca3

        # firing rates, used to decide the number of winners in kWTA
        self.firing_rate_ec = firing_rate_ec
        self.firing_rate_dg = firing_rate_dg
        self.firing_rate_ca3 = firing_rate_ca3

        # ============= setup Theano functions ==============
        m1 = T.fmatrix('m1')
        m2 = T.fmatrix('m2')
        result = m1.dot(m2)
        self.dot_product = theano.function([m1, m2], outputs=result)

        theta_ec = T.fmatrix('theta')
        f_theta = T.tanh(theta_ec/_EPSILON)
        self.transfer_function = theano.function([theta_ec], outputs=f_theta)

        # ============== ACTIVATION VALUES ==================
        input_values = np.zeros((1, dims[0])).astype(np.float32)
        for n_in in range(dims[0]):
            if np.random.random() < 0.5:
                input_values[0][n_in] = 1
            else:
                input_values[0][n_in] = -1
        self.input_values = theano.shared(name='input_values', value=input_values.astype(theano.config.floatX),
                                          borrow=True)

        ec_values = np.random.uniform(0, 1, (1, dims[1])).astype(np.float32)
        self.ec_values = theano.shared(name='ec_values', value=ec_values.astype(theano.config.floatX), borrow=True)

        dg_values = np.random.uniform(0, 1, (1, dims[2])).astype(np.float32)
        self.dg_values = theano.shared(name='dg_values', value=dg_values.astype(theano.config.floatX), borrow=True)

        ca3_values = np.random.uniform(0, 1, (1, dims[3])).astype(np.float32)
        self.ca3_values = theano.shared(name='ca3_values', value=ca3_values.astype(theano.config.floatX), borrow=True)

        output_values = np.zeros((1, dims[4])).astype(np.float32)
        for n_in in range(dims[4]):
            if np.random.random() < 0.5:
                output_values[0][n_in] = 1
            else:
                output_values[0][n_in] = -1
        self.output_values = theano.shared(name='output_values', value=output_values.astype(theano.config.floatX),
                                           borrow=True)

        # ============== WEIGHT MATRICES ===================
        input_ec_weights = np.ones((dims[0], dims[1])).astype(np.float32)
        # for each row in dims[0]: make 67 % of columns in dims[1] equal to 1, the rest 0:
        for row in range(dims[0]):
            for column in range(dims[1]):
                if np.random.random() < (1 - self.connection_rate_input_ec):
                    input_ec_weights[row][column] = 0
        self.in_ec_weights = theano.shared(name='in_ec_weights', value=input_ec_weights.astype(theano.config.floatX),
                                           borrow=True)

        ec_dg_weights = np.zeros((dims[1], dims[2])).astype(np.float32)
        # randomly assign about 25 % of the weights to a random connection weight
        for row in range(dims[1]):
            for col in range(dims[2]):
                if np.random.random() < self.PP:
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
                if np.random.random() < self.MF:
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
        # kWTA outputs:
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

        new_ec_values = T.fmatrix('new_ec_values')
        self.set_ec_values = theano.function([new_ec_values], outputs=None, updates=[(self.ec_values, new_ec_values)])

        y = T.iscalar('y')
        x = T.iscalar('x')
        val = T.fscalar('val')
        self.update_ec_dg_weights_value = theano.function(inputs=[y, x, val], updates=[(self.ec_dg_weights,
            T.set_subtensor(self.ec_dg_weights[y, x], val))])
        self.update_dg_ca3_weights_value = theano.function(inputs=[y, x, val], updates=[(self.dg_ca3_weights,
            T.set_subtensor(self.dg_ca3_weights[y, x], val))])
        self.update_ca3_ca3_weights_value = theano.function(inputs=[y, x, val], updates=[(self.ca3_ca3_weights,
            T.set_subtensor(self.ca3_ca3_weights[y, x], val))])
        # ===================================================

    # TODO: Theano-ize (parallelization). Use profiler?
    def neuronal_turnover_dg(self):
        # get beta %
        # for each of those neurons, initialize weights according to the percentage above.
        num_of_dg_neurons = self.dims[2]
        num_of_ca3_neurons = self.dims[3]
        num_of_ec_neurons = self.dims[1]

        num_of_neurons_to_be_turned_over = int(num_of_dg_neurons * _TURNOVER_RATE)
        for n in range(num_of_neurons_to_be_turned_over):
            # Note: These neurons may be drawn so that we get a more exact number of beta %. This implementation,
            #   however, introduces random fluctuations. Which might be beneficial?
            # this neuron is selected to have re-initialised its weights:
            random_dg_neuron_index = int(np.random.random() * num_of_dg_neurons)

            # from ec to dg:
            for ec_n in range(num_of_ec_neurons):
                if np.random.random() < self.PP:
                    self.update_ec_dg_weights_value(ec_n, random_dg_neuron_index, np.random.random())
                else:
                    self.update_ec_dg_weights_value(ec_n, random_dg_neuron_index, 0.0)
            # from dg to ca3:
            for ca3_neuron_index in range(num_of_ca3_neurons):
                if np.random.random() < self.MF:
                    self.update_dg_ca3_weights_value(random_dg_neuron_index, ca3_neuron_index, np.random.random())
                else:
                    self.update_dg_ca3_weights_value(random_dg_neuron_index, ca3_neuron_index, 0.0)

    def kWTA_ec(self):
        # kWTA EC:
        k_neurons_in_ec = int(self.dims[1] * self.firing_rate_ec)  # k determined by the firing rate

        # get a deep copy of the ec values in constant time:
        ec_act_vals = self.ec_values.get_value(borrow=False, return_internal_type=True)
        print "ec_act_vals:", ec_act_vals

        # for act_val_index in range(self.dims[1]):
        #     if T.ge(self.ec_values[1][act_val_index], k_th_largest_act_val_ec):
        #         self.ec_values[1][act_val_index].set_value(1)
        #     else:
        #         self.ec_values[1][act_val_index].set_value(0)

    # TODO: Check parallelism. Check further decentralization possibilities.
    def iter(self):
        # one iter for each part, such as:
        self.propagate_input_to_ec()
        self.kWTA_ec()

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

        # print "\nweights:"
        # print self.in_ec_weights.get_value()
        # print self.ec_dg_weights.get_value()
        # print self.ec_ca3_weights.get_value()
        # print self.ca3_ca3_weights.get_value()
        # print self.ca3_out_weights.get_value()


# testing code:

hpc = HPC([32, 240, 1600, 480, 32],
          0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
          0.10, 0.01, 0.04)  # firing rates: (ec, dg, ca3)
# sample IO:
# hpc.set_input(np.asarray([[1, 0, -1]]).astype(np.float32))
# hpc.set_output(np.asarray([[1, 0, -1]]).astype(np.float32))
# hpc.print_info()
for i in xrange(1):
    hpc.iter()
# hpc.neuronal_turnover_dg()
print "input:", hpc.input_values.get_value()