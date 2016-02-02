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