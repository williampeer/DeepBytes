from DNMM import *
from HPC import *

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
# All auto-associative pattern tests from the paper:
num_of_patterns_in_set = 2
number_of_sets = 5
learned_patterns = []
for i in range(number_of_sets):
    training_set = []
    for pattern in data_letters_capital[i*num_of_patterns_in_set:num_of_patterns_in_set + i*num_of_patterns_in_set]:
        io = [[]]
        for row in pattern:
            for el in row:
                io[0].append(el)
        new_array = np.asarray(io, dtype=np.float32)
        training_set.append([new_array, new_array])

    print "Performing neuronal turnover in DG for", hpc._turnover_rate * 100, "% of the neurons.."
    t0 = time.time()
    hpc.neuronal_turnover_dg()
    t1 = time.time()
    print "Neuronal turnover completed in", "{:7.3f}".format(t1-t0), "seconds."
    hpc_learn_patterns_wrapper(hpc, patterns=training_set, max_training_iterations=10)
    for pat in training_set:
        learned_patterns.append(pat)

hpc.show_image_from(np.asarray(next_experiment_im).astype(np.float32))
for learned_pattern in learned_patterns:
    hpc.setup_input(learned_pattern[0])
    hpc.recall()
    hpc.show_image_from(hpc.output_values.get_value())
# hpc_chaotic_recall_wrapper(hpc, display_images_of_intermediate_output=True, recall_iterations=300)