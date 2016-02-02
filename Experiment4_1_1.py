import numpy as np
import time
from data_capital import data_letters_capital
from DNMM import hpc_learn_patterns_wrapper, hpc_chaotic_recall_wrapper
from HPC import HPC
from Tools import show_image_from, set_contains_pattern

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
          0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha, alpha is 2 in 4.1.1

# next experiment output image:
next_experiment_im = [[-1, 1] * 24]
next_experiment_im[0].append(-1)


def experiment_4_1_1(training_set_size):
    # All auto-associative pattern tests from the paper:
    num_of_patterns_in_set = training_set_size
    number_of_sets = 5

    hippocampal_chaotic_recall_patterns = []

    for i in range(number_of_sets):
        training_set = []
        # Setup current training patterns:
        for pattern in data_letters_capital[i*num_of_patterns_in_set:num_of_patterns_in_set + i*num_of_patterns_in_set]:
            io = [[]]
            for row in pattern:
                for el in row:
                    io[0].append(el)
            new_array = np.asarray(io, dtype=np.float32)
            # print "Appending the following image..."
            # show_image_from(new_array)
            training_set.append([new_array, new_array])

        print "Performing neuronal turnover in DG for", hpc._turnover_rate * 100, "% of the neurons.."
        t0 = time.time()
        hpc.neuronal_turnover_dg()
        t1 = time.time()
        print "Neuronal turnover completed in", "{:7.3f}".format(t1-t0), "seconds."
        print "Learning patterns in training set..."
        hpc_learn_patterns_wrapper(hpc, patterns=training_set, max_training_iterations=15)

        # extract by chaotic recall:
        print "Recalling patterns for 300 time-steps by chaotic recall..."
        t2 = time.time()
        patterns_extracted_for_current_set = \
            hpc_chaotic_recall_wrapper(hpc, display_images_of_stable_output=False, recall_iterations=300)
        for pat in patterns_extracted_for_current_set:
            if not set_contains_pattern(hippocampal_chaotic_recall_patterns, pat):
                hippocampal_chaotic_recall_patterns.append(pat)  # append unique pattern
        t3 = time.time()
        print "Chaotic recall completed in", "{:8.3f}".format(t3-t2), "seconds, for t=300."

        # Use this to debug the current model:
        # learned_ctr = 0
        # for pat in training_set:
        #     hpc.setup_input(pat[0])
        #     print "Recalling pattern #", learned_ctr
        #     # show_image_from(np.asarray(next_experiment_im).astype(np.float32))
        #     hpc.recall()
        #     show_image_from(hpc.output_values.get_value())
        #     learned_ctr += 1

    # show_image_from(np.asarray(next_experiment_im).astype(np.float32))
    return hippocampal_chaotic_recall_patterns

hipp_chaotic_pats = experiment_4_1_1(2)
for recalled_pat in hipp_chaotic_pats:
    show_image_from(recalled_pat)