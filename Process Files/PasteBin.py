import numpy as np

io_dim = 49

# sample IO:
I = np.asarray([[1, -1, 1, -1, 1, -1, 1] * 7], dtype=np.float32)
O = np.asarray([[-1, 1, -1, 1, -1, 1, -1] * 7], dtype=np.float32)

rand_I = np.random.random((1, io_dim)).astype(np.float32) - 0.5 * np.ones((1, io_dim), dtype=np.float32)
rand_O = np.random.random((1, io_dim)).astype(np.float32) - 0.5 * np.ones((1, io_dim), dtype=np.float32)
for index in xrange(rand_I.shape[1]):
    if rand_I[0][index] < 0:
        rand_I[0][index] = -1
    else:
        rand_I[0][index] = 1
    if rand_O[0][index] < 0:
        rand_O[0][index] = -1
    else:
        rand_O[0][index] = 1


    # # TODO: Theano-ize (parallelization). Could use profiler.
    # def neuronal_turnover_dg(self):
    #     # get beta %
    #     # for each of those neurons, initialize weights according to the percentage above.
    #     num_of_dg_neurons = self.dims[2]
    #     num_of_ca3_neurons = self.dims[3]
    #     num_of_ec_neurons = self.dims[1]
    #
    #     num_of_neurons_to_be_turned_over = np.round(num_of_dg_neurons * self._turnover_rate).astype(np.int16)
    #     # np.random.seed(np.sqrt(time.time()).astype(np.int64))
    #     for n in range(num_of_neurons_to_be_turned_over):
    #         # Note: These neurons may be drawn so that we get a more exact number of beta %. This implementation,
    #         #   however, introduces random fluctuations. Which might be beneficial?
    #         # this neuron is selected to have re-initialised its weights:
    #         random_dg_neuron_index = np.round(np.random.random() * (num_of_dg_neurons-1)).astype(np.int16)
    #
    #         # from ec to dg:
    #         for ec_n in range(num_of_ec_neurons):
    #             if np.random.random() < self.PP:
    #                 self.update_ec_dg_weights_value(ec_n, random_dg_neuron_index, np.random.random())
    #             else:
    #                 self.update_ec_dg_weights_value(ec_n, random_dg_neuron_index, 0.0)
    #         # from dg to ca3:
    #         for ca3_neuron_index in range(num_of_ca3_neurons):
    #             if np.random.random() < self.MF:
    #                 self.update_dg_ca3_weights_value(random_dg_neuron_index, ca3_neuron_index, np.random.random())
    #             else:
    #                 self.update_dg_ca3_weights_value(random_dg_neuron_index, ca3_neuron_index, 0.0)


# def hpc_learn_patterns_iterations_hardcoded_wrapper(hpc, patterns):
#     print "Commencing learning of", len(patterns), "I/O patterns."
#     time_start_overall = time.time()
#     iter_ctr = 0
#     while iter_ctr < 2:
#         p_ctr = 0
#         for [input_pattern, output_pattern] in patterns:
#             # Neuronal turnover, setting input and output in the hpc network.
#             hpc.setup_pattern(input_pattern, output_pattern)
#
#             # one iteration of learning using Hebbian learning
#             hpc.learn()
#             p_ctr += 1
#
#         iter_ctr += 1
#     time_stop_overall = time.time()
#
#     print "Learned", len(patterns), "pattern-associations in ", iter_ctr, "iterations, which took" "{:7.3f}". \
#         format(time_stop_overall-time_start_overall), "seconds."


# NEURONAL TURNOVER SNIPPET:
# # Symbolically: DOESN'T WORK. WTF.
        # dg_res = T.fvector()
        # dg_num = T.iscalar()
        # ctr = T.iscalar()
        # _, updates_ec_dg = theano.scan(fn=self.neuronal_turnover_helper_ec_dg, outputs_info=ctr,
        #                                sequences=[dg_res, T.arange(dg_num)])
        # neuronal_turnover_ec_dg = theano.function([dg_res, dg_num, ctr], outputs=None, updates=updates_ec_dg)
        #
        # _, updates_dg_ca3 = theano.scan(self.neuronal_turnover_helper_dg_ca3, outputs_info=ctr,
        #                                 sequences=[dg_res, T.arange(dg_num)])
        # neuronal_turnover_dg_ca3 = theano.function([dg_res, dg_num, ctr], outputs=None, updates=updates_dg_ca3)

# def kWTA(self, values, firing_rate):
#         print "values:", values
#         # tuples = []
#         # index_ctr = 0
#         # for value in values[0]:
#         #     tuples.append((value, index_ctr))
#         #     index_ctr += 1
#
#         # kWTA EC:
#         k_neurons = np.floor(len(values[0]) * firing_rate).astype(np.int32)  # k determined by the firing rate
#         # k_neurons = int(len(values[0]) * firing_rate)  # k determined by the firing rate
#
#         sort_act_vals = theano.function([], outputs=T.sort(tuples))
#         act_vals_sorted = sort_act_vals()
#         k_th_largest_act_val = act_vals_sorted[len(values[0])-1 - k_neurons]  # TODO: Check that it is sorted in an ascending order.
#         print "act_vals_sorted:", act_vals_sorted
#         print "k_th_largest_act_val:", k_th_largest_act_val
#
#
#
#         # # TODO: Build hash-map. Draw random for same value 'til k nodes drawn.
#         # # TODO: Check if source for this bug stems from weights. Perhaps execute equations on paper?
#         #
#         # new_values = np.zeros_like(values, dtype=np.float32)
#         #
#         # for act_val_index in range(len(values[0])):
#         #     if values[0][act_val_index] > k_th_largest_act_val:
#         #         new_values[0][act_val_index] = 1
#         #     elif values[0][act_val_index] == k_th_largest_act_val:
#         #         if np.sum(new_values[0]) < k_neurons:
#         #             new_values[0][act_val_index] = 1
#         #         else:
#         #             return new_values
#         #
#         # return new_values


# def kWTA(self, values, f_r):
#         # print "values[0]", values[0]
#         values_length = len(values[0])
#         k = np.round(values_length * f_r).astype(np.int32)
#         values_sum = np.sum(values[0])
#         print "values_sum:", values_sum, "k:", k
#         # print "values_length:", values_length
#         # edge cases. note that the sum may be 0 or the length sometimes too without the edge case.
#         if values_sum == values_length or values_sum == 0:
#             print "equal sum to length or 0"
#             all_zero_or_one = True
#             for el in values[0]:
#                 if el != 0 and el != 1:
#                     # print "this el voiasdoipasd:", el
#                     all_zero_or_one = False
#                     print "all zero or one false"
#                     break
#             if all_zero_or_one:  # return random indices as on (1)
#                 return binomial_f(1, values_length, f_r)
#
#         sort_values = theano.function([], outputs=T.sort(values))
#         sorted_values = sort_values()
#         k_th_largest_value = sorted_values[0, values_length-k-1]
#
#         new_values = np.zeros_like(values)
#         k_ctr = 0
#         ind_ctr = 0
#         for el in values[0]:
#             if el > k_th_largest_value:
#                 new_values[0][ind_ctr] = 1
#                 k_ctr += 1
#             elif el == k_th_largest_value:
#                 if k_ctr < k:
#                     new_values[0][ind_ctr] = 1
#                     k_ctr += 1
#                 else:
#                     break
#             ind_ctr += 1
#
#         print "new_values:", new_values
#         print "np.sum(new_values):", np.sum(new_values)
#         return new_values