import numpy as np

from HPC import HPC
from NeocorticalNetwork import NeocorticalNetwork
from Experiments_4_x import experiment_4_x_1, experiment_4_x_2
from data_capital import data_letters_capital
from data_lowercase import data_letters_lowercase
import Tools

io_dim = 49

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

turnover_rate = 0.5  #(Tools.get_parameter_counter() % 18) * 0.02 + 0.32
weighting_dg = 25  # Tools.get_experiment_counter() % 26
_ASYNC_FLAG = False
_TURNOVER_MODE = 1  # 0 for between every new set. 1 for every set iteration.

# print "TRIAL #", trial, "turnover rate:", turnover_rate
# dims,
# connection_rate_input_ec, perforant_path, mossy_fibers,
#                  firing_rate_ec, firing_rate_dg, firing_rate_ca3,
#                  _gamma, _epsilon, _nu, _turnover_rate, _k_m, _k_r, _a_i, _alpha):
hpc = HPC([io_dim, 240, 1600, 480, io_dim],
          0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
          0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
          0.7, 100.0, 0.1, turnover_rate,  # gamma, epsilon, nu, turnover rate
          0.10, 0.95, 0.8, 2.0, weighting_dg,  # k_m, k_r, a_i, alpha. alpha is 2 in 4.1
          _ASYNC_FLAG=_ASYNC_FLAG, _TURNOVER_MODE=_TURNOVER_MODE)


# hpc.reset_hpc_module()
for i in range(20):
    for train_set_size_ctr in range(2, 6):
        Tools.append_line_to_log("INIT. EXPERIMENT MESSAGE: ASYNC-flag:" + str(_ASYNC_FLAG) + ". " +
                                 str(train_set_size_ctr) + "x5. " + "Turnover mode: " + str(_TURNOVER_MODE) +
                                 ". Turnover rate:"
                                 + str(turnover_rate) + ", DG-weighting: " + str(weighting_dg) + ".")

        tar_patts = []
        for p in training_patterns_associative[:5*train_set_size_ctr]:
            tar_patts.append(p[0])
        print tar_patts
        hipp_chaotic_pats, _ = experiment_4_x_1(hpc, train_set_size_ctr, training_patterns_associative)
        # write perfect recall rate to log:
        Tools.log_perfect_recall_rate(hipp_chaotic_pats, tar_patts)
        Tools.save_experiment_4_1_results(hpc, hipp_chaotic_pats, "train_set_size_"+str(train_set_size_ctr)+"_exp_1"+
                                          "turnover_rate_" + str(turnover_rate) +
                                          "weighting_" + str(hpc._weighting_dg))

        # ann = SimpleNeocorticalNetwork(io_dim, 30, io_dim, 0.01, 0.9)

        # print "Starting experiment 4_2..."
        # This also saves the experiment_4_x_1 results!
        # information_vector = experiment_4_x_2(hpc, ann, train_set_size_ctr,
        #                                       training_patterns_associative[:5 * train_set_size_ctr])
        # print "Saving the results."
        # Tools.save_experiment_4_2_results(information_vector, "train_set_size_" + str(train_set_size_ctr) +
        #                                   "_exp_2_")

        # For now, this is the ONLY place where the counter is incremented.
        Tools.increment_experiment_counter()
