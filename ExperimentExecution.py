import numpy as np

from HPC import HPC
from SimpleNeocorticalNetwork import SimpleNeocorticalNetwork
from Experiments_4_x import experiment_4_x_1, experiment_4_x_2
from data_capital import data_letters_capital
from data_lowercase import data_letters_lowercase
from Tools import show_image_from, save_experiment_4_1_results, save_experiment_4_2_results, save_images_from

# Hippocampal module
io_dim = 49

ann = SimpleNeocorticalNetwork(io_dim, 30, io_dim, 0.01, 0.9)

training_patterns_associative = []
# Setup all training patterns:
for letter_data in data_letters_capital:
    io = [[]]
    for row in letter_data:
        for el in row:
            io[0].append(el)
    new_array = np.asarray(io, dtype=np.float32)
    training_patterns_associative.append([new_array, new_array])

training_patterns_heterogeneous = []
letter_ctr = 0
for letter_data in data_letters_lowercase:
    io_lowercase = [[]]
    for row in letter_data:
        for el in row:
            io_lowercase[0].append(el)
    lowercase_letter = np.asarray(io_lowercase, dtype=np.float32)
    uppercase_letter = training_patterns_associative[letter_ctr][0]
    training_patterns_heterogeneous.append([uppercase_letter, lowercase_letter])

for i in range(1, 6):
    # dims,
    # connection_rate_input_ec, perforant_path, mossy_fibers,
    #                  firing_rate_ec, firing_rate_dg, firing_rate_ca3,
    #                  _gamma, _epsilon, _nu, _turnover_rate, _k_m, _k_r, _a_i, _alpha):
    hpc = HPC([io_dim, 240, 1600, 480, io_dim],
              0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
              0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
              0.7, 1.0, 0.1, 0.5,  # gamma, epsilon, nu, turnover rate
              0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha. alpha is 2 in 4.1
    hipp_chaotic_pats = experiment_4_x_1(hpc, i, training_patterns_associative)
    save_experiment_4_1_results(hpc, hipp_chaotic_pats, "test_"+str(i)+"_patterns_")
    save_images_from(hipp_chaotic_pats)

    for recalled_pat in hipp_chaotic_pats:
        print "Displayed chaotically recalled image!"
        show_image_from(recalled_pat)

    # print "Starting experiment 4_2..."
    # information_vector = experiment_4_x_2(hpc, ann, i, training_patterns_associative)
    # print "Saving the results."
    # save_experiment_4_2_results(information_vector, "test_"+str(i)+"_4_2_")
    # print "Saving the images..."
    # save_images_from(information_vector[0]+information_vector[1])
