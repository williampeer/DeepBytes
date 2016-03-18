import theano
import theano.tensor as T
import numpy as np
from Tools import binomial_f, uniform_f, show_image_from

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
        ec_dg_weights = binomial_f(dims[1], dims[2], self.PP) * uniform_f(dims[1], dims[2])
        self.ec_dg_weights = theano.shared(name='ec_dg_weights', value=ec_dg_weights.astype(theano.config.floatX),
                                           borrow=True)

        # randomly assign all weights between the EC and CA3
        ec_ca3_weights = uniform_f(dims[1], dims[3])
        self.ec_ca3_weights = theano.shared(name='ec_ca3_weights', value=ec_ca3_weights.astype(theano.config.floatX),
                                            borrow=True)

        # randomly assign about 4 % of the weights to random connection weights
        dg_ca3_weights = binomial_f(dims[2], dims[3], self.MF) * uniform_f(dims[2], dims[3])  # elemwise
        self.dg_ca3_weights = theano.shared(name='dg_ca3_weights', value=dg_ca3_weights.astype(theano.config.floatX),
                                            borrow=True)

        # randomly assign 100 % of the weights between CA3 and CA3
        ca3_ca3_weights = uniform_f(dims[3], dims[3])
        self.ca3_ca3_weights = theano.shared(name='ca3_ca3_weights', value=ca3_ca3_weights.astype(theano.config.floatX),
                                             borrow=True)

        # random weight assignment, full connection rate CA3-out
        ca3_output_weights = uniform_f(dims[3], dims[4])
        self.ca3_out_weights = theano.shared(name='ca3_out_weights',
                                             value=ca3_output_weights.astype(theano.config.floatX), borrow=True)

        # Vector constant
        a_i_arr = _a_i * np.ones_like(ca3_values, dtype=np.float32)
        self._a_i = theano.shared(name='_a_i', value=a_i_arr.astype(theano.config.floatX))

        #
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

        X_i = T.fvector()
        x_j = T.fscalar()
        weights_column = T.fvector()

        weights_column_update_unconstrained = self._gamma * weights_column + x_j * X_i
        self.unconstrained_hebbian_equation = theano.function([X_i, x_j, weights_column],
                                                            outputs=weights_column_update_unconstrained)

        weights_column_update_constrained = weights_column + self._nu * x_j * (X_i + x_j * weights_column)
        self.constrained_hebbian_equation = theano.function([X_i, x_j, weights_column],
                                                              outputs=weights_column_update_constrained)

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
        next_activation_values_out = T.tanh(c_ca3_vals.dot(c_ca3_out_Ws) / self._epsilon)
        self.fire_ca3_out = theano.function([c_ca3_vals, c_ca3_out_Ws], updates=[(self.output_values,
                                                                                  next_activation_values_out)])

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

        self.set_output = theano.function([new_activation_values], outputs=None,
                                          updates=[(self.output_values, new_activation_values)])

        new_weights = T.fmatrix('new_weights')
        self.update_input_ec_weights = theano.function(inputs=[new_weights], updates=[(self.in_ec_weights, new_weights)])

        weights_update_vector = T.fvector('weight_column_update')
        index = T.iscalar('column_index')
        self.update_ec_dg_weights_column = theano.function([index, weights_update_vector],
            updates={self.ec_dg_weights: T.set_subtensor(self.ec_dg_weights[:, index], weights_update_vector)})

        self.update_ec_ca3_weights_column = theano.function([index, weights_update_vector],
            updates={self.ec_ca3_weights: T.set_subtensor(self.ec_ca3_weights[:, index], weights_update_vector)})

        self.update_dg_ca3_weights_column = theano.function([index, weights_update_vector],
            updates={self.dg_ca3_weights: T.set_subtensor(self.dg_ca3_weights[:, index], weights_update_vector)})

        self.update_ca3_ca3_weights_column = theano.function([index, weights_update_vector],
            updates={self.ca3_ca3_weights: T.set_subtensor(self.ca3_ca3_weights[:, index], weights_update_vector)})

        self.update_ca3_out_weights_column= theano.function([index, weights_update_vector],
            updates={self.ca3_out_weights: T.set_subtensor(self.ca3_out_weights[:, index], weights_update_vector)})

        self.update_dg_ca3_weights_row = theano.function([index, weights_update_vector],
            updates={self.dg_ca3_weights: T.set_subtensor(self.dg_ca3_weights[index, :], weights_update_vector)})
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
        weights_vector = uniform_f(1, self.dims[1]) * weights_row_connection_rate_factor
        self.update_ec_dg_weights_column(column_index, weights_vector[0])

    def neuronal_turnover_helper_dg_ca3(self, row_index):
        # DG neuron connections are rewired.
        # for every neuron in dg, rewire its weights to all neurons of ca3
        weights_row_connection_rate_factor = binomial_f(1, self.dims[3], self.MF)
        # multiply with random weights:
        weights_vector = uniform_f(1, self.dims[3]) * weights_row_connection_rate_factor
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

    def fire_in_ec_wrapper(self):
        self.propagate_input_to_ec(self.input_values.get_value(return_internal_type=True),
                                   self.in_ec_weights.get_value(return_internal_type=True))
        # kWTA for EC firing:
        self.set_ec_values(
                self.kWTA(self.ec_values.get_value(return_internal_type=True), self.firing_rate_ec))  # in-memory copy

    def fire_ec_dg_wrapper(self):
        # fire EC to DG
        self.fire_ec_dg(self.ec_values.get_value(return_internal_type=True),
                        self.ec_dg_weights.get_value(return_internal_type=True))
        # kWTA
        self.set_dg_values(
                self.kWTA(self.dg_values.get_value(return_internal_type=True), self.firing_rate_dg))  # in-memory copy

    def fire_to_ca3_wrapper(self):
        # fire EC to CA3, DG to CA3, and CA3 to CA3
        current_ec_vals = self.ec_values.get_value(return_internal_type=True)
        current_ec_ca3_Ws = self.ec_ca3_weights.get_value(return_internal_type=True)
        current_dg_vals = self.dg_values.get_value(return_internal_type=True)
        current_dg_ca3_Ws = self.dg_ca3_weights.get_value(return_internal_type=True)
        current_ca3_vals = self.ca3_values.get_value(borrow=False)
        current_ca3_ca3_Ws = self.ca3_ca3_weights.get_value(return_internal_type=True)
        current_nu_ca3 = self.nu_ca3.get_value(return_internal_type=True)
        current_zeta_ca3 = self.zeta_ca3.get_value(return_internal_type=True)
        self.fire_all_to_ca3(current_ec_vals, current_ec_ca3_Ws, current_dg_vals, current_dg_ca3_Ws, current_ca3_vals,
                             current_ca3_ca3_Ws, current_nu_ca3, current_zeta_ca3)
        # kWTA
        self.set_ca3_values(
                self.kWTA(self.ca3_values.get_value(return_internal_type=True), self.firing_rate_ca3))  # in-memory copy

    def fire_ca3_out_wrapper(self):
        # fire CA3 to output
        self.fire_ca3_out(self.ca3_values.get_value(return_internal_type=True),
                          self.ca3_out_weights.get_value(return_internal_type=True))

        # Bipolar output:
        self.set_output(self.get_bipolar_in_out_values(self.output_values.get_value(return_internal_type=True)))

    def wire_ec_dg_wrapper(self):
        activation_values = self.dg_values.get_value()
        for neuron_index in range(activation_values.shape[1]):
            if activation_values[0][neuron_index] == 1:
                # weights update: update column i
                X_i = self.ec_values.get_value()[0]
                x_j = activation_values[0][neuron_index]
                weight_column = self.ec_dg_weights.get_value()[:, neuron_index]

                self.update_ec_dg_weights_column(
                        weight_column_update=self.constrained_hebbian_equation(X_i, x_j, weight_column),
                        column_index=neuron_index)

    def wire_ec_ca3_wrapper(self):
        activation_values = self.ca3_values.get_value()
        for neuron_index in range(activation_values.shape[1]):
            if activation_values[0][neuron_index] == 1:
                # weights update: update column i
                X_i = self.ec_values.get_value()[0]
                x_j = activation_values[0][neuron_index]
                weight_column = self.ec_ca3_weights.get_value()[:, neuron_index]
                self.update_ec_ca3_weights_column(
                        weight_column_update=self.constrained_hebbian_equation(X_i, x_j, weight_column),
                        column_index=neuron_index)

    def wire_dg_ca3_wrapper(self):
        activation_values = self.ca3_values.get_value()
        for neuron_index in range(activation_values.shape[1]):
            if activation_values[0][neuron_index] == 1:
                # weights update: update column i
                X_i = self.dg_values.get_value()[0]
                x_j = activation_values[0][neuron_index]
                weight_column = self.dg_ca3_weights.get_value()[:, neuron_index]
                self.update_dg_ca3_weights_column(
                        weight_column_update=self.constrained_hebbian_equation(X_i, x_j, weight_column),
                        column_index=neuron_index)

    def wire_ca3_ca3_wrapper(self):
        activation_values = self.ca3_values.get_value()
        for neuron_index in range(activation_values.shape[1]):
            if activation_values[0][neuron_index] == 1:
                # weights update: update column i
                X_i = self.ca3_values.get_value()[0]
                x_j = activation_values[0][neuron_index]
                weight_column = self.ca3_ca3_weights.get_value()[:, neuron_index]
                self.update_ca3_ca3_weights_column(
                        weight_column_update=self.unconstrained_hebbian_equation(X_i, x_j, weight_column),
                        column_index=neuron_index)

    def wire_ca3_out_wrapper(self):
        activation_values = self.output_values.get_value()
        for neuron_index in range(activation_values.shape[1]):
            if activation_values[0][neuron_index] == 1:
                # weights update: update column i
                X_i = self.ca3_values.get_value()[0]
                x_j = activation_values[0][neuron_index]
                weight_column = self.ca3_out_weights.get_value()[:, neuron_index]
                self.update_ca3_out_weights_column(
                        weight_column_update=self.unconstrained_hebbian_equation(X_i, x_j, weight_column),
                        column_index=neuron_index)

    def learn(self, I, O):
        self.set_input(I)
        self.set_output(O)

        self.fire_in_ec_wrapper()
        self.fire_ec_dg_wrapper()
        self.fire_to_ca3_wrapper()

        self.wire_ec_dg_wrapper()
        self.wire_ec_ca3_wrapper()
        self.wire_dg_ca3_wrapper()
        self.wire_ca3_ca3_wrapper()
        # Without firing!
        self.wire_ca3_out_wrapper()

        # self.print_activation_values_and_weights()
        # self.print_activation_values_sum()
        # self.print_min_max_weights()
        # print "self.nu_ca3.get_value():", self.nu_ca3.get_value()
        # print "self.zeta_ca3.get_value():", self.zeta_ca3.get_value()

    def recall(self, I):
        self.set_input(I)
        self.fire_in_ec_wrapper()
        self.fire_to_ca3_wrapper()
        self.fire_ca3_out_wrapper()

    def recall_using_current_input(self):
        self.fire_to_ca3_wrapper()
        self.fire_ca3_out_wrapper()

    def recall_random(self):
        I = binomial_f(self.input_values.shape[0], self.input_values.shape[1], 0.5) * 2 - \
            np.ones(shape=self.input_values.shape, dtype=np.float32)
        self.recall(I)

    # recall until output .. unchanged for three iterations
    def recall_until_stability_criteria(self, should_display_image, max_iterations):
        out_now = np.copy(self.output_values.get_value(borrow=False))
        out_t_minus_1 = np.zeros_like(out_now, dtype=np.float32)

        found_stable_output = False
        ctr = 0
        while not found_stable_output and ctr < max_iterations:
            out_t_minus_2 = np.copy(out_t_minus_1)
            out_t_minus_1 = np.copy(out_now)

            self.recall_using_current_input()
            out_now = np.copy(self.output_values.get_value(borrow=False))

            # Check if output has been unchanged for three time-steps
            found_stable_output = True
            for out_y in xrange(out_now.shape[1]):
                if not (out_t_minus_2[0][out_y] == out_t_minus_1[0][out_y] == out_now[0][out_y]):
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
        sum=0
        for el in self.input_values.get_value()[0]:
            sum+=el
        print "sum input:", sum
        sum=0
        for el in self.ec_values.get_value()[0]:
            sum+=el
        print "sum ec:", sum
        sum=0
        for el in self.dg_values.get_value()[0]:
            sum+=el
        print "sum dg:", sum

        sum=0
        for el in self.ca3_values.get_value()[0]:
            sum+=el
        print "sum ca3:", sum
        if sum==0:
            print "NO ACTIVE NEURONS IN --CA3--."
        sum=0
        for el in self.output_values.get_value()[0]:
            sum+=el
        print "sum output:", sum

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