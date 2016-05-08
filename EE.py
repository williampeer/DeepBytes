import numpy as np

from HPC import HPC
from NeocorticalNetwork import NeocorticalNetwork
import Experiments_4_x
from DataWrapper import training_patterns_associative
from DataWrapper import training_patterns_heterogeneous
import Tools

io_dim = 49

turnover_rate = 0.5  #(Tools.get_parameter_counter() % 18) * 0.02 + 0.32
weighting_dg = 25  # Tools.get_experiment_counter() % 26
_ASYNC_FLAG = True
_TURNOVER_MODE = 0  # 0 for between every new set. 1 for every set iteration.

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

for i in range(20):
    for train_set_size_ctr in range(2, 6):
        hpc.reset_hpc_module()

        tar_patts = []
        for p in training_patterns_associative[:5*train_set_size_ctr]:
            tar_patts.append(p[1])

        ann = NeocorticalNetwork(io_dim, 30, io_dim, 0.01, 0.9)

        print "Starting experiment 4_2..."
        # This also saves the experiment_4_x_1 results!
        # information_vector = experiment_4_x_2(hpc, ann, train_set_size_ctr,
        #                                       training_patterns_associative[:5 * train_set_size_ctr])
        # information_vector = Experiments_4_x.experiment_4_2_neo_version(ann, train_set_size_ctr,
        #                                                                 training_patterns_associative
        #                                                                 [:5 * train_set_size_ctr])
        information_vector = Experiments_4_x.experiment_4_2_neo_pseudorehearsal_with_chaotic_patterns(
                hpc, ann, train_set_size_ctr, training_patterns_associative[:5 * train_set_size_ctr])

        print "Saving the results."
        Tools.save_experiment_4_2_results(information_vector, "train_set_size_" + str(train_set_size_ctr) +
                                          "_exp_2_")

        # For now, this is the ONLY place where the counter is incremented.
        Tools.increment_experiment_counter()
