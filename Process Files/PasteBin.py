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


        # TURNOVER W/ SCAN FAILED ATTEMPT:
        # get beta %
        # for each of those neurons, initialize weights according to the percentage above.

        # Execution:
        # num_of_dg_neurons = self.dims[2]
        # dg_neuron_selection = binomial_f(1, num_of_dg_neurons, self._turnover_rate)
        # neuron_index = 0
        # target_indices = []
        # for dg_sel in dg_neuron_selection[0]:
        #     if dg_sel == 1:
        #         target_indices.append(neuron_index)
        #     neuron_index += 1
        #
        # indices = T.ivector('indices')
        # random_weights_sequence = T.fvectors('random_weights_sequence')
        # ec_dg_weights = T.fmatrix('ec_dg_weights')
        # new_ec_dg_weights = T.fmatrix('new_ec_dg_weights')
        # ec_dg_results, ec_dg_updates = theano.scan(fn=self.return_weight_column,
        #                                            outputs_info=new_ec_dg_weights,
        #                                            sequences=[indices, random_weights_sequence],
        #                                            non_sequences=[ec_dg_weights])
        # perform_turnover_ec_dg = theano.function([indices, random_weights_sequence, ec_dg_weights],
        #                                          outputs=ec_dg_results)
        #
        # #
        # column_length = self.ec_dg_weights.get_value().shape[0]
        # index_sequence = np.asarray(target_indices, dtype=np.int32)
        # # print "index seq:", index_sequence
        # new_column_weights = random_f(index_sequence.shape[1], column_length) * binomial_f(index_sequence,
        #                                                                                    column_length, self.PP)
        #
        # print perform_turnover_ec_dg(index_sequence, new_column_weights, self.ec_dg_weights.get_value())

    # column_index = T.iscalar("column_index")
    #     weight_column = T.fvector("weight_column")
    #     weight_matrix = T.fmatrix("weight_matrix")
    #     self.return_weight_column = theano.function([column_index, weight_column, weight_matrix],
    #                                                 outputs=T.set_subtensor(
    #                                                         weight_matrix[:, column_index], weight_column))


# returns: [ experiment: [set_size, #sets, DGW, 5x tuples: [iters_to_convergence, distinct_patterns_recalled]]]
def get_data_from_log_file(filename):
    log_file = file(filename, 'r')

    contents = log_file.read()
    log_file.close()

    lines = contents.split('\n')
    all_data = []
    for experiment_ctr in range(len(lines)/11):
        experiment_data = []

        sep = lines[11*experiment_ctr].split('x')
        set_size = int(sep[0][len(sep[0])-1])
        training_sets = int(sep[1][0])

        words = lines[experiment_ctr * 11].split()
        dg_weighting = words[-1][:len(words[-1])-1]

        experiment_data.append(set_size)
        experiment_data.append(training_sets)
        experiment_data.append(float(dg_weighting))

        for i in range(5):
            convergence_line_words = lines[11 * experiment_ctr + 1 + 2 * i].split()
            patterns_recalled_line_words = lines[11 * experiment_ctr + 2 + 2 * i].split()

            iterations_before_convergence = int(convergence_line_words[2])
            distinct_patterns_recalled = int(patterns_recalled_line_words[1])
            experiment_data.append([iterations_before_convergence, distinct_patterns_recalled])

        all_data.append(experiment_data)

    return all_data

# def generate_pseudopattern_II_hpc_outputs(dim, hpc_extracted_pseudopatterns, reverse_P, set_size):
#     extracted_set_size = len(hpc_extracted_pseudopatterns)
#     pseudopatterns_II = []
#     pseudopattern_ctr = 0
#     while pseudopattern_ctr < set_size:
#         pattern = hpc_extracted_pseudopatterns[pseudopattern_ctr % extracted_set_size]
#         # q=1-p because we're flipping the sign of the ones that are not flipped.
#         reverse_vector = Tools.binomial_f(1, dim, (1-reverse_P))
#         reverse_vector = reverse_vector * 2 - np.ones_like(reverse_vector)
#         pseudopatterns_II.append(pattern * reverse_vector)
#         pseudopattern_ctr += 1
#     return pseudopatterns_II