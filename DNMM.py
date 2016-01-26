import time
from HPC import *
from SimpleNeocorticalNetwork import *
from data_capital import *

def hpc_learn_patterns_wrapper(hpc, patterns, max_training_iterations):
    print "Commencing learning of", len(patterns), "I/O patterns."
    time_start_overall = time.time()
    iter_ctr = 0
    learned_all = False
    while not learned_all and iter_ctr < max_training_iterations:
        p_ctr = 0
        for [input_pattern, output_pattern] in patterns:
            # Neuronal turnover, setting input and output in the hpc network.
            setup_start = time.time()
            hpc.setup_pattern(input_pattern, output_pattern)
            setup_end = time.time()
            print "Setup took:", "{:0.3f}".format(setup_end-setup_start), "seconds."

            # one iteration of learning using Hebbian learning
            time_before = time.time()
            hpc.learn()
            # hpc.print_activation_values_sum()
            time_after = time.time()
            print "Iterated over pattern", p_ctr, "in", \
                "{:0.3f}".format(time_after - time_before), "seconds."
            p_ctr += 1

        learned_all = True
        print "Attempting to recall patterns..."
        for pattern_index in xrange(len(patterns)):
            print "Recalling pattern #", pattern_index
            hpc.setup_input(patterns[pattern_index][0])
            hpc.recall()
            out_values = hpc.output_values.get_value()[0]
            cur_p = patterns[pattern_index][0][0]
            # print "outvals:", out_values
            # print "curp", cur_p
            for el_index in xrange(len(cur_p)):
                if out_values[el_index] != cur_p[el_index]:
                    learned_all = False
                    print "Patterns are not yet successfully learned. Learning more..."
                    break
            if not learned_all:
                break

        iter_ctr += 1
    time_stop_overall = time.time()

    print "Learned", len(patterns), "pattern-associations in ", iter_ctr, "iterations, which took" "{:0.3f}". \
        format(time_stop_overall-time_start_overall), "seconds."

def hpc_chaotic_recall_wrapper(hpc, display_images_of_intermediate_output, recall_iterations):
    time_the_beginning_of_time = time.time()
    time_before = time.time()
    cur_iters = 0
    new_random_input = np.ones_like(hpc.input_values.get_value(), dtype=np.float32)
    np.random.seed()
    for rand_in_index in xrange(new_random_input.shape[1]):
        if np.random.random() < 0.5:
            new_random_input[0][rand_in_index] = -1
    hpc.setup_input(new_random_input)
    while cur_iters < recall_iterations:
        cur_iters += hpc.recall_until_stability_criteria(should_display_image=display_images_of_intermediate_output,
                                                         max_iterations=recall_iterations-cur_iters)
        time_after = time.time()
        prop_time_until_stable = time_after - time_before

        print "Propagation time until stability:", "{:0.3f}".format(prop_time_until_stable), "seconds."
        print "t =", cur_iters
        time_before = time.time()
    print "Total chaotic recall time:", "{:6.3f}".format(time.time()-time_the_beginning_of_time), "seconds."

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

training_set = []
num_to_learn = 3
for pattern in data_letters_capital[:num_to_learn]:
    io = [[]]
    for row in pattern:
        for el in row:
            io[0].append(el)
    new_array = np.asarray(io, dtype=np.float32)
    training_set.append([new_array, new_array])
# patterns.reverse()

hpc_learn_patterns_wrapper(hpc, patterns=training_set, max_training_iterations=10)

print "Recalling all learned patterns:"
for i in xrange(len(training_set[:num_to_learn])):
    print "Pattern #", i
    hpc.setup_input(training_set[i][0])
    hpc.recall()
    hpc.show_image_from(hpc.output_values.get_value())

hpc_chaotic_recall_wrapper(hpc, display_images_of_intermediate_output=True, recall_iterations=50)
# hpc.print_info()


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