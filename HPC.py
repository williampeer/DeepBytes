import theano
import theano.tensor as T
import numpy as np
from Tools import binomial_f, uniform_f, random_f, show_image_from

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
        self._alpha = _alpha

        # ============== ACTIVATION VALUES ==================
        input_values = np.zeros((1, dims[0]), dtype=np.float32)
        self.input_values = theano.shared(name='input_values', value=input_values.astype(theano.config.floatX),
                                          borrow=True)

        # in order to create fmatrices, we need to call random.random, and not zeros(1, N).
        ec_values = uniform_f(1, dims[1])
        self.ec_values = theano.shared(name='ec_values', value=ec_values.astype(theano.config.floatX), borrow=True)

        dg_values = uniform_f(1, dims[2])
        self.dg_values = theano.shared(name='dg_values', value=dg_values.astype(theano.config.floatX), borrow=True)

        ca3_values = uniform_f(1, dims[3])
        self.ca3_values = theano.shared(name='ca3_values', value=ca3_values.astype(theano.config.floatX), borrow=True)
        # prev_ca3_values = np.zeros_like(ca3_values, dtype=np.float32)
        # self.prev_ca3_values = theano.shared(name='prev_ca3_values', value=prev_ca3_values.astype(theano.config.floatX),
        #                                      borrow=True)

        output_values = np.zeros((1, dims[4])).astype(np.float32)
        self.output_values = theano.shared(name='output_values', value=output_values.astype(theano.config.floatX),
                                           borrow=True)

        # =========== CA3 CHAOTIC NEURONS SETUP ============
        nu_ca3 = np.zeros_like(ca3_values, dtype=np.float32)
        self.nu_ca3 = theano.shared(name='nu_ca3', value=nu_ca3.astype(theano.config.floatX), borrow=True)

        zeta_ca3 = np.zeros_like(ca3_values, dtype=np.float32)
        self.zeta_ca3 = theano.shared(name='zeta_ca3', value=zeta_ca3.astype(theano.config.floatX), borrow=True)

        # ============== WEIGHT MATRICES ===================
        input_ec_weights = binomial_f(dims[0], dims[1], self.connection_rate_input_ec)
        self.in_ec_weights = theano.shared(name='in_ec_weights', value=input_ec_weights.astype(theano.config.floatX),
                                           borrow=True)

        # randomly assign about 25 % of the weights to a random connection weight
        # ec_dg_weights = binomial_f(dims[1], dims[2], self.PP) * uniform_f(dims[1], dims[2])
        ec_dg_weights = binomial_f(dims[1], dims[2], self.PP) * random_f(dims[1], dims[2])
        self.ec_dg_weights = theano.shared(name='ec_dg_weights', value=ec_dg_weights.astype(theano.config.floatX),
                                           borrow=True)

        # randomly assign all weights between the EC and CA3
        # ec_ca3_weights = uniform_f(dims[1], dims[3])
        ec_ca3_weights = random_f(dims[1], dims[3])
        self.ec_ca3_weights = theano.shared(name='ec_ca3_weights', value=ec_ca3_weights.astype(theano.config.floatX),
                                            borrow=True)

        # randomly assign about 4 % of the weights to random connection weights
        # dg_ca3_weights = binomial_f(dims[2], dims[3], self.MF) * uniform_f(dims[2], dims[3])  # elemwise
        dg_ca3_weights = binomial_f(dims[2], dims[3], self.MF) * random_f(dims[2], dims[3])  # elemwise
        self.dg_ca3_weights = theano.shared(name='dg_ca3_weights', value=dg_ca3_weights.astype(theano.config.floatX),
                                            borrow=True)

        # randomly assign 100 % of the weights between CA3 and CA3
        # ca3_ca3_weights = uniform_f(dims[3], dims[3])
        ca3_ca3_weights = random_f(dims[3], dims[3])
        self.ca3_ca3_weights = theano.shared(name='ca3_ca3_weights', value=ca3_ca3_weights.astype(theano.config.floatX),
                                             borrow=True)

        # random weight assignment, full connection rate CA3-out
        # ca3_output_weights = uniform_f(dims[3], dims[4])
        ca3_output_weights = random_f(dims[3], dims[4])
        self.ca3_out_weights = theano.shared(name='ca3_out_weights',
                                             value=ca3_output_weights.astype(theano.config.floatX), borrow=True)

        # Vector constant
        a_i_arr = _a_i * np.ones_like(ca3_values, dtype=np.float32)
        self._a_i = theano.shared(name='_a_i', value=a_i_arr.astype(theano.config.floatX))

        # ============== HEBBIAN LEARNING ==================
        # Input:
        # Apparently, weights are 0 or 1 between the input and EC and constant after initialization
        local_in_vals = T.fmatrix()
        local_in_ec_Ws = T.fmatrix()
        next_activation_values_ec = T.tanh(local_in_vals.dot(local_in_ec_Ws) / self._epsilon)
        self.propagate_input_to_ec = theano.function([local_in_vals, local_in_ec_Ws], outputs=None,
                                                     updates=[(self.ec_values, next_activation_values_ec)])

        # ================= CONSTRAINED ====================
        # kWTA outputs:
        local_ec_vals = T.fmatrix()
        local_ec_dg_Ws = T.fmatrix()
        next_activation_values_dg = T.tanh(local_ec_vals.dot(local_ec_dg_Ws) / self._epsilon)
        self.fire_ec_dg = theano.function([local_ec_vals, local_ec_dg_Ws], outputs=None,
                                          updates=[(self.dg_values, next_activation_values_dg)])

        # wire after kWTA for this layer
        u_prev_reshaped_transposed = T.fmatrix('u_prev_reshaped_transposed')
        u_next_reshaped = T.fmatrix('u_next_reshaped')
        Ws_prev_next = T.fmatrix('Ws_prev_next')
        # Element-wise operations. w_13_next = w_13 + nu u_3(u_1-u_3 w_13).
        next_Ws = Ws_prev_next + self._nu * u_next_reshaped * (u_prev_reshaped_transposed.T - u_next_reshaped * Ws_prev_next)
        self.wire_ec_dg = theano.function([u_prev_reshaped_transposed, u_next_reshaped, Ws_prev_next],
                                          updates=[(self.ec_dg_weights, next_Ws)])

        # ============= CA3 =============
        # "c_"-prefix to avoid shadowing.
        c_ec_vals = T.fmatrix()
        c_ec_ca3_Ws = T.fmatrix()
        c_dg_vals = T.fmatrix()
        c_dg_ca3_Ws = T.fmatrix()
        c_ca3_vals = T.fmatrix()
        c_ca3_ca3_Ws = T.fmatrix()
        c_nu_ca3 = T.fmatrix()
        c_zeta_ca3 = T.fmatrix()

        ca3_input_sum = c_ec_vals.dot(c_ec_ca3_Ws) + c_dg_vals.dot(c_dg_ca3_Ws) + c_ca3_vals.dot(c_ca3_ca3_Ws)
        nu_ca3 = self._k_m * c_nu_ca3 + ca3_input_sum
        zeta_ca3 = self._k_r * c_zeta_ca3 - self._alpha * c_ca3_vals + self._a_i
        next_activation_values_ca3 = T.tanh((nu_ca3 + zeta_ca3) / self._epsilon)
        self.fire_all_to_ca3 = theano.function([c_ec_vals, c_ec_ca3_Ws, c_dg_vals, c_dg_ca3_Ws, c_ca3_vals,
                                                c_ca3_ca3_Ws, c_nu_ca3, c_zeta_ca3],
                                               updates=[(self.ca3_values, next_activation_values_ca3),
                                                        (self.nu_ca3, nu_ca3), (self.zeta_ca3, zeta_ca3)])

        # after kWTA:
        self.wire_ec_to_ca3 = theano.function([u_prev_reshaped_transposed, u_next_reshaped, Ws_prev_next], updates=[
            (self.ec_ca3_weights, next_Ws)])
        self.wire_dg_to_ca3 = theano.function([u_prev_reshaped_transposed, u_next_reshaped, Ws_prev_next], updates=[
            (self.dg_ca3_weights, next_Ws)])

        local_ca3_ca3_Ws = T.fmatrix()
        local_ca3_vals = T.fmatrix()
        local_prev_ca3_vals = T.fmatrix()
        next_ca3_ca3_weights = self._gamma * local_ca3_ca3_Ws + T.transpose(local_ca3_vals).dot(local_prev_ca3_vals)
        self.wire_ca3_to_ca3 = theano.function([local_ca3_vals, local_prev_ca3_vals, local_ca3_ca3_Ws],
                                               updates=[(self.ca3_ca3_weights, next_ca3_ca3_weights)])

        # without learning:
        no_learning_ca3_input_sum = c_ec_vals.dot(c_ec_ca3_Ws) + c_ca3_vals.dot(c_ca3_ca3_Ws)
        no_learning_nu_ca3 = self._k_m * c_nu_ca3 + no_learning_ca3_input_sum
        no_learning_zeta_ca3 = self._k_r * c_zeta_ca3 - self._alpha * c_ca3_vals + self._a_i
        no_learning_next_act_vals_ca3 = T.tanh((no_learning_nu_ca3 + no_learning_zeta_ca3) / self._epsilon)
        self.fire_to_ca3_no_learning = theano.function([c_ec_vals, c_ec_ca3_Ws, c_ca3_vals, c_ca3_ca3_Ws, c_nu_ca3,
                                                        c_zeta_ca3],
                                                       updates=[(self.ca3_values, no_learning_next_act_vals_ca3),
                                                                (self.nu_ca3, no_learning_nu_ca3),
                                                                (self.zeta_ca3, no_learning_zeta_ca3)])

        # ===================================================
        # Output:
        c_ca3_out_Ws = T.fmatrix()
        c_out_values = T.fmatrix()
        next_activation_values_out = T.tanh(c_ca3_vals.dot(c_ca3_out_Ws) / self._epsilon)
        self.fire_ca3_out = theano.function([c_ca3_vals, c_ca3_out_Ws], updates=[(self.output_values,
                                                                                  next_activation_values_out)])
        next_ca3_out_weights = self._gamma * c_ca3_out_Ws + T.transpose(c_ca3_vals).dot(c_out_values)
        self.wire_ca3_out = theano.function([c_ca3_vals, c_out_values, c_ca3_out_Ws],
                                            updates=[(self.ca3_out_weights, next_ca3_out_weights)])

        # =================== SETTERS =======================
        new_activation_values = T.fmatrix('new_activation_values')
        self.set_input = theano.function([new_activation_values], outputs=None,
                                         updates=[(self.input_values, new_activation_values)])

        self.set_ec_values = theano.function([new_activation_values], outputs=None,
                                             updates=[(self.ec_values, new_activation_values)])

        self.set_dg_values = theano.function([new_activation_values], outputs=None,
                                             updates=[(self.dg_values, new_activation_values)])

        self.set_ca3_values = theano.function([new_activation_values], outputs=None,
                                              updates=[(self.ca3_values, new_activation_values)])

        # self.set_prev_ca3_values = theano.function([new_activation_values], outputs=None,
        #                                            updates=[(self.prev_ca3_values, new_activation_values)])

        self.set_output = theano.function([new_activation_values], outputs=None,
                                          updates=[(self.output_values, new_activation_values)])

        new_weights = T.fmatrix('new_weights')
        self.update_input_ec_weights = theano.function(inputs=[new_weights], updates=[(self.in_ec_weights, new_weights)])

        new_weights_vector = T.fvector('new_weights_vector')
        index_x_or_y = T.iscalar()
        self.update_ec_dg_weights_column = theano.function([index_x_or_y, new_weights_vector],
            updates={self.ec_dg_weights: T.set_subtensor(self.ec_dg_weights[:, index_x_or_y], new_weights_vector)})
        self.update_dg_ca3_weights_row = theano.function([index_x_or_y, new_weights_vector],
            updates={self.dg_ca3_weights: T.set_subtensor(self.dg_ca3_weights[index_x_or_y, :], new_weights_vector)})

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

    # Partly optimized neuronal turnover. Not sure how to make the scan operations work.
    def neuronal_turnover_dg(self):
        # get beta %
        # for each of those neurons, initialize weights according to the percentage above.

        # Execution:
        num_of_dg_neurons = self.dims[2]
        dg_neuron_selection = binomial_f(1, num_of_dg_neurons, self._turnover_rate)
        neuron_index = 0
        for dg_sel in dg_neuron_selection[0]:
            if dg_sel == 1:
                self.neuronal_turnover_helper_ec_dg(neuron_index)
                self.neuronal_turnover_helper_dg_ca3(neuron_index)
            neuron_index += 1

    def neuronal_turnover_helper_ec_dg(self, column_index):
        # DG neuron connections are rewired.
        # for every neuron in ec, rewire its weights to this neuron - that means ONE row in the weights matrix!
        weights_row_connection_rate_factor = binomial_f(1, self.dims[1], self.PP)
        # multiply with random weights:
        # weights_vector = uniform_f(1, self.dims[1]) * weights_row_connection_rate_factor
        weights_vector = random_f(1, self.dims[1]) * weights_row_connection_rate_factor
        self.update_ec_dg_weights_column(column_index, weights_vector[0])

    def neuronal_turnover_helper_dg_ca3(self, row_index):
        # DG neuron connections are rewired.
        # for every neuron in dg, rewire its weights to all neurons of ca3
        weights_row_connection_rate_factor = binomial_f(1, self.dims[3], self.MF)
        # multiply with random weights:
        # weights_vector = uniform_f(1, self.dims[3]) * weights_row_connection_rate_factor
        weights_vector = random_f(1, self.dims[3]) * weights_row_connection_rate_factor
        self.update_dg_ca3_weights_row(row_index, weights_vector[0])

    def re_wire_fixed_input_to_ec_weights(self):
        input_ec_weights = binomial_f(self.dims[0], self.dims[1], 0.67)
        self.update_input_ec_weights(input_ec_weights)

    # Returns a vector with the corresponding output, i.e. the k largest values as 1, the rest 0.
    def kWTA(self, values, f_r):
        # print "values[0]", values[0]
        values_length = len(values[0])
        k = np.round(values_length * f_r).astype(np.int32)

        sort_values_f = theano.function([], outputs=T.sort(values))
        sorted_values = sort_values_f()
        k_th_largest_value = sorted_values[0][values_length-k-1]

        mask_vector = k_th_largest_value * np.ones_like(values)
        result = (values >= mask_vector).astype(np.float32)

        sum_result = np.sum(result)
        if sum_result > k:
            # iterate through result vector
            excess_elements_count = (sum_result - k).astype(np.int32)
            ind_map = []
            ind_ctr = 0
            for el in result[0]:
                if el == 1:
                    ind_map.append(ind_ctr)
                ind_ctr += 1
            map_len = len(ind_map)-1
            for i in range(excess_elements_count):
                random_ind = np.round(map_len * np.random.random()).astype(np.int32)
                flip_ind = ind_map[random_ind]
                result[0][flip_ind] = 0
                ind_map.remove(flip_ind)
                map_len -= 1

        return result

    # TODO: Check parallelism. Check further decentralization possibilities.
    def learn(self):
        # one iteration for each layer/HPC-part
        # self.internal_recall()  # sets the t-1 nu- and zeta-values for recalling the current letter.

        # fire EC to DG
        self.fire_ec_dg(self.ec_values.get_value(return_internal_type=True),
                        self.ec_dg_weights.get_value(return_internal_type=True))
        # kWTA
        self.set_dg_values(
                self.kWTA(self.dg_values.get_value(return_internal_type=True), self.firing_rate_dg))  # in-memory copy

        # wire EC to DG
        n_rows_for_u_next = self.ec_values.get_value(return_internal_type=True).shape[1]
        n_cols_for_u_prev = self.dg_values.get_value(return_internal_type=True).shape[1]
        u_next_for_elemwise_ops = [self.dg_values.get_value(return_internal_type=True)[0]] * n_rows_for_u_next
        u_prev_for_elemwise_ops_transposed = [self.ec_values.get_value(return_internal_type=True)[0]] * n_cols_for_u_prev

        self.wire_ec_dg(u_prev_for_elemwise_ops_transposed, u_next_for_elemwise_ops,
                        self.ec_dg_weights.get_value(return_internal_type=True))

        # fire EC to CA3, DG to CA3, and CA3 to CA3
        l_ec_vals = self.ec_values.get_value(return_internal_type=True)
        l_ec_ca3_Ws = self.ec_ca3_weights.get_value(return_internal_type=True)
        l_dg_vals = self.dg_values.get_value(return_internal_type=True)
        l_dg_ca3_Ws = self.dg_ca3_weights.get_value(return_internal_type=True)
        l_ca3_vals = self.ca3_values.get_value(borrow=False)
        l_ca3_ca3_Ws = self.ca3_ca3_weights.get_value(return_internal_type=True)
        l_nu_ca3 = self.nu_ca3.get_value(return_internal_type=True)
        l_zeta_ca3 = self.zeta_ca3.get_value(return_internal_type=True)
        self.fire_all_to_ca3(l_ec_vals, l_ec_ca3_Ws, l_dg_vals, l_dg_ca3_Ws, l_ca3_vals, l_ca3_ca3_Ws, l_nu_ca3, l_zeta_ca3)
        # kWTA
        self.set_ca3_values(
                self.kWTA(self.ca3_values.get_value(return_internal_type=True), self.firing_rate_ca3))  # in-memory copy

        # wire EC to CA3
        n_rows = self.ec_values.get_value(return_internal_type=True).shape[1]
        n_cols = self.ca3_values.get_value(return_internal_type=True).shape[1]
        u_next_for_elemwise_ops = [self.ca3_values.get_value(return_internal_type=True)[0]] * n_rows
        u_prev_for_elemwise_ops_transposed = [self.ec_values.get_value(return_internal_type=True)[0]] * n_cols
        self.wire_ec_to_ca3(u_prev_for_elemwise_ops_transposed, u_next_for_elemwise_ops,
                            self.ec_ca3_weights.get_value(return_internal_type=True))

        # wire DG to CA3
        n_rows = self.dg_values.get_value(return_internal_type=True).shape[1]
        u_next_for_elemwise_ops = [self.ca3_values.get_value(return_internal_type=True)[0]] * n_rows
        u_prev_for_elemwise_ops_transposed = [self.dg_values.get_value(return_internal_type=True)[0]] * n_cols
        self.wire_dg_to_ca3(u_prev_for_elemwise_ops_transposed, u_next_for_elemwise_ops,
                            self.dg_ca3_weights.get_value(return_internal_type=True))

        # wire CA3 to CA3
        self.wire_ca3_to_ca3(self.ca3_values.get_value(return_internal_type=True),
                             l_ca3_vals,
                             self.ca3_ca3_weights.get_value(return_internal_type=True))
        self.wire_ca3_out(self.ca3_values.get_value(return_internal_type=True),
                          self.output_values.get_value(return_internal_type=True),
                          self.ca3_out_weights.get_value(return_internal_type=True))

        # self.print_activation_values_and_weights()
        # self.print_activation_values_sum()
        # self.print_min_max_weights()
        # print "self.nu_ca3.get_value():", self.nu_ca3.get_value()
        # print "self.zeta_ca3.get_value():", self.zeta_ca3.get_value()

    def setup_input(self, input_pattern):
        # self.re_wire_fixed_input_to_ec_weights()
        self.set_input(input_pattern)

        self.propagate_input_to_ec(self.input_values.get_value(return_internal_type=True),
                                   self.in_ec_weights.get_value(return_internal_type=True))
        # kWTA for EC firing:
        self.set_ec_values(
                self.kWTA(self.ec_values.get_value(return_internal_type=True), self.firing_rate_ec))  # in-memory copy

    def setup_pattern(self, input_pattern, output_pattern):
        # self.neuronal_turnover_dg()
        self.setup_input(input_pattern)
        self.set_output(output_pattern)

    # Method used to update the previous nu- and zeta-values with the current I/O pattern before weight updates.
    def internal_recall(self):
        # Fire EC to CA3, CA3 to CA3
        self.fire_to_ca3_no_learning(self.ec_values.get_value(return_internal_type=True),
                                     self.ec_ca3_weights.get_value(return_internal_type=True),
                                     self.ca3_values.get_value(return_internal_type=True),
                                     self.ca3_ca3_weights.get_value(return_internal_type=True),
                                     self.nu_ca3.get_value(return_internal_type=True),
                                     self.zeta_ca3.get_value(return_internal_type=True))
        # kWTA CA3
        self.set_ca3_values(
                self.kWTA(self.ca3_values.get_value(return_internal_type=True), self.firing_rate_ca3))  # in-memory

        # fire CA3 to output
        self.fire_ca3_out(self.ca3_values.get_value(return_internal_type=True),
                          self.ca3_out_weights.get_value(return_internal_type=True))

        # Bipolar output:
        self.set_output(self.get_bipolar_in_out_values(self.output_values.get_value(return_internal_type=True)))

    def recall(self):
        # Fire EC to CA3, CA3 to CA3
        self.fire_to_ca3_no_learning(self.ec_values.get_value(return_internal_type=True),
                                     self.ec_ca3_weights.get_value(return_internal_type=True),
                                     self.ca3_values.get_value(return_internal_type=True),
                                     self.ca3_ca3_weights.get_value(return_internal_type=True),
                                     self.nu_ca3.get_value(return_internal_type=True),
                                     self.zeta_ca3.get_value(return_internal_type=True))
        # kWTA CA3
        self.set_ca3_values(
                self.kWTA(self.ca3_values.get_value(return_internal_type=True), self.firing_rate_ca3))  # in-memory

        # fire CA3 to output
        self.fire_ca3_out(self.ca3_values.get_value(return_internal_type=True),
                          self.ca3_out_weights.get_value(return_internal_type=True))

        # Bipolar output:
        self.set_output(self.get_bipolar_in_out_values(self.output_values.get_value(return_internal_type=True)))
        # show_image_from(self.output_values.get_value())

    def recall_until_stability_criteria(self, should_display_image, max_iterations):  # recall until output unchanged three iterations
        out_now = np.copy(self.output_values.get_value(borrow=False))
        out_min_1 = np.zeros_like(out_now, dtype=np.float32)
        found_stable_output = False
        ctr = 0
        while not found_stable_output and ctr < max_iterations:
            out_min_2 = np.copy(out_min_1)
            out_min_1 = np.copy(out_now)

            # Attempt to set a random input for every iteration:
            self.recall()
            out_now = np.copy(self.output_values.get_value(borrow=False))
            found_stable_output = True
            for out_y in xrange(out_now.shape[1]):
                if not (out_min_2[0][out_y] == out_min_1[0][out_y] == out_now[0][out_y]):
                    found_stable_output = False
                    break
            ctr += 1
        if should_display_image and found_stable_output:
            show_image_from(out_now=out_now)

        print "Reached stability or max. #iterations during chaotic recall after", ctr, "iterations."
        return [ctr, found_stable_output, out_now]

    def get_bipolar_in_out_values(self, values):
        # new_values = values + 0.000001 * np.ones_like(values, dtype=np.float32)
        # return new_values / np.abs(new_values)

        new_values = np.ones_like(values, dtype=np.float32)
        for value_index in xrange(values.shape[1]):
            if values[0][value_index] < 0:
                new_values[0][value_index] = -1
        return new_values

    # ================================================ DEBUG ===============================================
    def print_activation_values_sum(self):
        print
        ctr=0
        for el in self.input_values.get_value()[0]:
            ctr+=el
        print "sum input:", ctr
        ctr=0
        for el in self.ec_values.get_value()[0]:
            ctr+=el
        print "sum ec:", ctr
        ctr=0
        for el in self.dg_values.get_value()[0]:
            ctr+=el
        print "sum dg:", ctr

        ctr=0
        for el in self.ca3_values.get_value()[0]:
            ctr+=el
        print "sum ca3:", ctr
        if ctr==0:
            print "NO ACTIVE NEURONS IN --CA3--."
        ctr=0
        for el in self.output_values.get_value()[0]:
            ctr+=el
        print "sum output:", ctr

    def print_activation_values(self):
        print "\nprinting activation values:"
        print "in:", self.input_values.get_value()
        print "ec:", self.ec_values.get_value()
        print "dg:", self.dg_values.get_value()
        print "ca3:", self.ca3_values.get_value()
        print "out:", self.output_values.get_value()

    def print_last_halves_of_activation_values_sums(self):
        print "\nprinting activation values:"
        print "sum in, last 50 %:", np.sum(self.input_values.get_value()[0][len(self.input_values.get_value()[0])/2:])
        print "sum ec, last 50 %:", np.sum(self.ec_values.get_value()[0][len(self.ec_values.get_value()[0])/2:])
        print "sum dg, last 50 %:", np.sum(self.dg_values.get_value()[0][len(self.dg_values.get_value()[0])/2:])
        print "sum ca3, last 50 %:", np.sum(self.ca3_values.get_value()[0][len(self.ca3_values.get_value()[0])/2:])
        print "sum out, last 50 %:", np.sum(self.output_values.get_value()[0][len(self.output_values.get_value()[0])/2:])

    def print_activation_values_and_weights(self):
        print "\nprinting activation values:"
        print "in:", self.input_values.get_value()
        print "ec:", self.ec_values.get_value()
        print "dg:", self.dg_values.get_value()
        print "ca3:", self.ca3_values.get_value()
        print "out:", self.output_values.get_value()

        print "\nweights:"
        print "in-ec:", self.in_ec_weights.get_value()
        print "ec-dg:", self.ec_dg_weights.get_value()
        print "ec-ca3:", self.ec_ca3_weights.get_value()
        print "dg-ca3:", self.dg_ca3_weights.get_value()
        print "CA3-CA3:", self.ca3_ca3_weights.get_value()
        print "ca3-out:", self.ca3_out_weights.get_value()

    def print_min_max_weights(self):
        print "min-max weights in ca3-ca3 and ca3-out"
        min_ca3_ca3 = np.min(self.ca3_ca3_weights.get_value())
        max_ca3_ca3 = np.max(self.ca3_ca3_weights.get_value())
        min_ca3_out = np.min(self.ca3_out_weights.get_value())
        max_ca3_out = np.max(self.ca3_out_weights.get_value())
        print "min, max ca3-ca3:", min_ca3_ca3, max_ca3_ca3
        print "min, max ca3-out:", min_ca3_out, max_ca3_out

    def print_ca3_info(self):
        print "ca3:", self.ca3_values.get_value()
        dot_result1 = self.ec_values.get_value().dot(self.ec_ca3_weights.get_value())
        dot_result2 = self.dg_values.get_value().dot(self.dg_ca3_weights.get_value())
        dot_result3 = self.ca3_values.get_value().dot(self.ca3_ca3_weights.get_value())
        print "ec-ca3:", dot_result1
        print "dg-ca3:", dot_result2
        print "ca3-ca3:", dot_result3
        dot_sum = dot_result1 + dot_result2 + dot_result3
        print "sum:", dot_sum
        print "kWTA:", self.kWTA(dot_sum, 0.04)
        print "firing rate ca3:", self.firing_rate_ca3

    def test_pydotprint(self):
        theano.printing.pydotprint(self.fire_all_to_ca3, outfile="/hpc_pydotprint_test.png",
                                   var_with_name_simple=True)