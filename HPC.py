import theano
import theano.tensor as T
import numpy as np
import time

# Note: Ensure float32 for GPU-usage. Use the profiler to analyse GPU-usage.
theano.config.floatX = 'float32'


# dims: neuron layer sizes
# gamma: forgetting factor
# epsilon: steepness parameter (used in transfer function)
# nu: learning rate
# k_m, k_r: damping factors of refractoriness
# a_i: external input parameter
# alpha: scaling factor for refractoriness
class HPC:
    def __init__(self, dims, connection_rate_input_ec, perforant_path, mossy_fibers,
                 firing_rate_ec, firing_rate_dg, firing_rate_ca3,
                 _gamma, _epsilon, _nu, _turnover_rate, _k_m, _k_r, _a_i, _alpha):

        # =================== PARAMETERS ====================
        self.dims = dims  # numbers of neurons in the different layers
        self.connection_rate_input_ec = connection_rate_input_ec
        self.PP = perforant_path  # connection_rate_ec_dg
        self.MF = mossy_fibers  # connection_rate_dg_ca3

        # firing rates, used to decide the number of winners in kWTA
        self.firing_rate_ec = firing_rate_ec
        self.firing_rate_dg = firing_rate_dg
        self.firing_rate_ca3 = firing_rate_ca3

        # constants
        self._gamma = _gamma
        self._epsilon = _epsilon
        self._nu = _nu
        self._turnover_rate = _turnover_rate
        self._k_m = _k_m
        self._k_r = _k_r
        self._a_i = _a_i
        self._alpha = _alpha

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

        # =========== CA3 CHAOTIC NEURONS SETUP ============
        nu_ca3 = np.zeros_like(ca3_values, dtype=np.float32)
        self.nu_ca3 = theano.shared(name='nu_ca3', value=nu_ca3.astype(theano.config.floatX), borrow=True)

        zeta_ca3 = np.zeros_like(ca3_values, dtype=np.float32)
        self.zeta_ca3 = theano.shared(name='zeta_ca3', value=zeta_ca3.astype(theano.config.floatX), borrow=True)

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
                    sign = 1
                    if np.random.random() < 0.5:
                        sign = -1
                    ec_dg_weights[row][col] = sign * np.random.random()
        self.ec_dg_weights = theano.shared(name='ec_dg_weights', value=ec_dg_weights.astype(theano.config.floatX),
                                           borrow=True)

        # randomly assign all weights between the EC and CA3
        ec_ca3_weights = np.random.uniform(-1, 1, (dims[1], dims[3])).astype(np.float32)
        self.ec_ca3_weights = theano.shared(name='ec_ca3_weights', value=ec_ca3_weights.astype(theano.config.floatX),
                                            borrow=True)

        dg_ca3_weights = np.zeros((dims[2], dims[3])).astype(np.float32)
        # randomly assign about 4 % of the weights to random connection weights
        for row in range(dims[2]):
            for col in range(dims[3]):
                if np.random.random() < self.MF:
                    sign = 1
                    if np.random.random() < 0.5:
                        sign = -1
                    dg_ca3_weights[row][col] = sign * np.random.random()
        self.dg_ca3_weights = theano.shared(name='dg_ca3_weights', value=dg_ca3_weights.astype(theano.config.floatX),
                                            borrow=True)

        # randomly assign 100 % of the weights
        ca3_ca3_weights = np.random.uniform(-1, 1, (dims[3], dims[3])).astype(np.float32)
        self.ca3_ca3_weights = theano.shared(name='ca3_ca3_weights', value=ca3_ca3_weights.astype(theano.config.floatX),
                                             borrow=True)

        ca3_output_weights = np.random.uniform(-1, 1, (dims[3], dims[4])).astype(np.float32)
        self.ca3_out_weights = theano.shared(name='ca3_out_weights',
                                             value=ca3_output_weights.astype(theano.config.floatX), borrow=True)

        # ============== HEBBIAN LEARNING ==================
        # Input:
        # Apparently, weights are 0 or 1 between the input and EC and constant after initialization
        next_activation_values_ec = T.tanh(self.input_values.dot(self.in_ec_weights) / self._epsilon)
        self.propagate_input_to_ec = theano.function([], outputs=None, updates=[(self.ec_values,
                                                                                 next_activation_values_ec)])

        # ================= CONSTRAINED ====================
        # kWTA outputs:
        next_activation_values_dg = T.tanh(self.ec_values.dot(self.ec_dg_weights) / self._epsilon)
        next_ec_dg_weights = self.ec_dg_weights + self._nu * (self.ec_values - (self.dg_values.dot(
                T.transpose(self.ec_dg_weights)))).T.dot(next_activation_values_dg)
        self.fire_ec_dg = theano.function([], outputs=None, updates=[(self.dg_values, next_activation_values_dg)])
        self.wire_ec_dg = theano.function([], outputs=None, updates=[(self.ec_dg_weights, next_ec_dg_weights)])

        # ============= CA3 =============
        ca3_input_sum = self.ec_values.dot(self.ec_ca3_weights) + \
            self.dg_values.dot(self.dg_ca3_weights) + self.ca3_values.dot(self.ca3_ca3_weights)

        nu_ca3 = self._k_m * self.nu_ca3 + ca3_input_sum
        zeta_ca3 = self._k_r * self.zeta_ca3 - self._alpha * self.ca3_values + self._a_i
        next_activation_values_ca3 = T.tanh((nu_ca3 + zeta_ca3) / self._epsilon)

        next_ec_ca3_weights = self.ec_ca3_weights + self._nu * (self.ec_values - next_activation_values_ca3.dot(
                T.transpose(self.ec_ca3_weights))).T.dot(next_activation_values_ca3)

        next_dg_ca3_weights = self.dg_ca3_weights + self._nu * (self.dg_values - next_activation_values_ca3.dot(
                T.transpose(self.dg_ca3_weights))).T.dot(next_activation_values_ca3)

        next_ca3_ca3_weights = self._gamma * self.ca3_ca3_weights + \
                               T.transpose(next_activation_values_ca3).dot(self.ca3_values)
        self.fire_all_to_ca3 = theano.function([], outputs=None, updates=[(self.ca3_values, next_activation_values_ca3)])
        self.wire_ca3 = theano.function([], outputs=None, updates=[(self.ec_ca3_weights, next_ec_ca3_weights),
                                                                   (self.dg_ca3_weights, next_dg_ca3_weights),
                                                                   (self.ca3_ca3_weights, next_ca3_ca3_weights),
                                                                   (self.nu_ca3, nu_ca3), (self.zeta_ca3, zeta_ca3)])

        # without learning:
        no_learning_ca3_input_sum = self.ec_values.dot(self.ec_ca3_weights) + self.ca3_values.dot(self.ca3_ca3_weights)
        no_learning_nu_ca3 = self._k_m * self.nu_ca3 + no_learning_ca3_input_sum
        no_learning_zeta_ca3 = self._k_r * self.zeta_ca3 - self._alpha * self.ca3_values + self._a_i
        no_learning_next_act_vals_ca3 = T.tanh((no_learning_nu_ca3 + no_learning_zeta_ca3) / self._epsilon)
        self.fire_to_ca3_no_learning = theano.function([], outputs=None, updates=[
            (self.ca3_values, no_learning_next_act_vals_ca3), (self.nu_ca3, no_learning_nu_ca3),
            (self.zeta_ca3, no_learning_zeta_ca3)])

        # ===================================================
        # Output:
        next_activation_values_out = T.tanh(self.ca3_values.dot(self.ca3_out_weights) / self._epsilon)
        self.fire_ca3_out = theano.function([], outputs=None, updates=[(self.output_values, next_activation_values_out)])
        next_ca3_out_weights = self._gamma * self.ca3_out_weights + T.transpose(self.ca3_values).\
            dot(self.output_values.get_value())
        self.wire_ca3_out = theano.function([], outputs=None, updates=[(self.ca3_out_weights, next_ca3_out_weights)])

        # =================== SETTERS =======================
        new_input = T.fmatrix('new_input')
        self.set_input = theano.function([new_input], outputs=None,
                                         updates=[(self.input_values, new_input)])

        new_ec_values = T.fmatrix('new_ec_values')
        self.set_ec_values = theano.function([new_ec_values], outputs=None, updates=[(self.ec_values, new_ec_values)])

        new_dg_values = T.fmatrix('new_dg_values')
        self.set_dg_values = theano.function([new_dg_values], outputs=None, updates=[(self.dg_values, new_dg_values)])

        new_ca3_values = T.fmatrix('new_ca3_values')
        self.set_ca3_values = theano.function([new_ca3_values], outputs=None, updates=[(self.ca3_values, new_ca3_values)])

        new_output = T.fmatrix('new_output')
        self.set_output = theano.function([new_output], outputs=None, updates=[(self.output_values, new_output)])

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

        num_of_neurons_to_be_turned_over = int(num_of_dg_neurons * self._turnover_rate)
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

    #
    # Returns a vector with the corresponding output, i.e. the k largest values as 1, the rest 0.
    def kWTA(self, values, firing_rate):
        # kWTA EC:
        k_neurons = int(len(values[0]) * firing_rate)  # k determined by the firing rate

        sort_act_vals = theano.function([], outputs=T.sort(values))
        act_vals_sorted = sort_act_vals()
        k_th_largest_act_val = act_vals_sorted[0, len(values[0]) - k_neurons-1]

        new_values = np.zeros_like(values, dtype=np.float32)

        for act_val_index in range(len(values[0])):
            if values[0, act_val_index] >= k_th_largest_act_val:
                new_values[0, act_val_index] = 1

        return new_values

    # TODO: Check parallelism. Check further decentralization possibilities.
    def iter(self):
        # one iteration for each layer/HPC-part
        self.propagate_input_to_ec()
        # kWTA for EC firing:
        self.set_ec_values(
                self.kWTA(self.ec_values.get_value(borrow=False, return_internal_type=True), self.firing_rate_ec))  # deep in-memory copy
        self.fire_ec_dg()
        self.set_dg_values(
                self.kWTA(self.dg_values.get_value(borrow=False, return_internal_type=True), self.firing_rate_dg))  # deep in-memory copy
        self.wire_ec_dg()
        self.fire_all_to_ca3()
        self.set_ca3_values(
                self.kWTA(self.ca3_values.get_value(borrow=False, return_internal_type=True), self.firing_rate_ca3))  # deep in-memory copy
        self.wire_ca3()
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
          0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
          0.7, 0.1, 1, 0.5,  # gamma, epsilon, nu, turnover rate
          0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha, alpha is 2, 4, and 5 in different experiments in Hattori (2014)
# sample IO:
# hpc.set_input(np.asarray([[1, 0, -1]]).astype(np.float32))
# hpc.set_output(np.asarray([[1, 0, -1]]).astype(np.float32))
# hpc.print_info()
time_before = time.time()
for i in xrange(10):
    hpc.iter()
time_after = time.time()
hpc.print_info()
print "Execution time: ", time_after - time_before
# hpc.neuronal_turnover_dg()
# print "input:", hpc.input_values.get_value()