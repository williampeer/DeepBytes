import time
from HPC import *
from SimpleNeocorticalNetwork import *
from data_capital import *

def hpc_learn_patterns_wrapper(hpc, patterns, training_iterations):
    print "Commencing learning of", len(patterns), "I/O patterns."
    time_start_overall = time.time()
    for [input_pattern, output_pattern] in patterns:
        # Neuronal turnover, setting input and output in the hpc network.
        setup_start = time.time()
        hpc.setup_pattern(input_pattern, output_pattern)
        setup_end = time.time()
        print "Setup took:", "{:10.4f}".format(setup_end-setup_start), "seconds."

        time_before = time.time()
        for i in xrange(training_iterations):
            # one iteration of learning using Hebbian learning
            hpc.learn()
            hpc.print_activation_values_sum()
        time_after = time.time()
        train_time = time_after - time_before
        print "Training time: ", "{:10.4f}".format(train_time), "seconds. Trained for", training_iterations, "iteration(s)."
    time_stop_overall = time.time()
    print "Learned", len(patterns), "pattern-associations in ", "{:10.4f}".format(time_stop_overall-time_start_overall), "seconds."

def hpc_chaotic_recall_wrapper(hpc, display_images_of_intermediate_output, recall_iterations):
    time_before = time.time()
    cur_iters = 0
    while cur_iters < recall_iterations:
        new_random_input = np.ones_like(hpc.input_values.get_value(), dtype=np.float32)
        np.random.seed(np.sqrt(time.time()).astype(np.int64))
        for rand_in_index in xrange(new_random_input.shape[1]):
            if np.random.random() < 0.5:
                new_random_input[0][rand_in_index] = -1
        hpc.setup_input(new_random_input)
        cur_iters += hpc.recall_until_stability_criteria(should_display_image=display_images_of_intermediate_output,
                                                         max_iterations=recall_iterations-cur_iters)
        time_after = time.time()
        prop_time_until_stable = time_after - time_before

        print "Propagation time until stability:", "{:0.3f}".format(prop_time_until_stable), "seconds."
        print "t =", cur_iters
        time_before = time.time()

# ==================== TESTING CODE: ======================
# Hippocampal module
io_dim = 49
# dims,
# connection_rate_input_ec, perforant_path, mossy_fibers,
#                  firing_rate_ec, firing_rate_dg, firing_rate_ca3,
#                  _gamma, _epsilon, _nu, _turnover_rate, _k_m, _k_r, _a_i, _alpha):
hpc = HPC([io_dim, 240, 1600, 480, io_dim],
          0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
          0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
          0.7, 1.0, 0.1, 0.5,  # gamma, epsilon, nu, turnover rate
          0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha, alpha is 2, 4, and 5 in different experiments in Hattori (2014)

patterns = []
num_to_learn = 1
for pattern in data_letters_capital[:num_to_learn]:
    io = [[]]
    for row in pattern:
        for el in row:
            io[0].append(el)
    new_array = np.asarray(io, dtype=np.float32)
    patterns.append([new_array, new_array])
# patterns.reverse()
hpc.print_info()
hpc_learn_patterns_wrapper(hpc, patterns=patterns, training_iterations=1)
# hpc.print_info()

# hpc_chaotic_recall_wrapper(hpc, display_images_of_intermediate_output=False)

print "Recalling all learned patterns:"
for i in xrange(len(patterns[:num_to_learn])):
    print "Pattern #", i
    hpc.setup_input(patterns[i][0])
    hpc.recall()
    hpc.show_image_from(hpc.output_values.get_value())
    hpc.print_activation_values_sum()

hpc.print_info()

# Neocortical module:
# ann = SimpleNeocorticalNetwork(32, 50, 32, 0.85, 0.01)
#
# # a = np.random.random((1, 32)).astype(np.float32)
# # b = -1 * np.random.random((1, 32)).astype(np.float32)
# a = np.asarray([[0.1, 0.2] * 16], dtype=np.float32)
# b = np.asarray([[-0.2, -0.4] * 16], dtype=np.float32)
#
# iopair = [a, b]
#
# print "target output:", b
# for i in range(20000):
#     ann.train([iopair])
# print ann.in_h_Ws.get_value()
# print ann.h_out_Ws.get_value()
# ann.print_layers()