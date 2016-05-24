from HPC import HPC
from NeocorticalNetwork import NeocorticalNetwork
import Experiments_4_x
from DataWrapper import training_patterns_associative
# from DataWrapper import training_patterns_heterogeneous
import Tools

io_dim = 49

turnover_rate = 0.50  # (Tools.get_parameter_counter() % 18) * 0.02 + 0.32
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

# ============ Config. 1: ============ async, 0.50, tm 0, dg 25, local
for i in range(30):
    weighting_dg = i
    hpc._weighting_dg = i
    # remaining iterations:
    for j in range(10):
        for train_set_size_ctr in range(2, 6):
            hpc.reset_hpc_module()

            tar_patts = []
            for p in training_patterns_associative[:5*train_set_size_ctr]:
                tar_patts.append(p[1])

            print "Starting experiment 4.1, HPC chaotic recall i iters and HPC pseudopatterns..."
            hippocampal_chaotic_recall_patterns, random_ins = Experiments_4_x.experiment_4_x_1(
                    hpc, train_set_size_ctr, training_patterns_associative[:5 * train_set_size_ctr])

            Tools.save_experiment_4_1_results(hpc, random_ins, hippocampal_chaotic_recall_patterns, tar_patts,
                                              "async bug fixed", train_set_size_ctr)

            # For now, this is the ONLY place where the counter is incremented.
            Tools.increment_experiment_counter()

# ============ Config. 1: ============ async, 0.04, tm 1, dg 25, local
hpc._TURNOVER_MODE = 1
hpc._turnover_rate = 0.04
for i in range(30):
    weighting_dg = i
    hpc._weighting_dg = i
    # remaining iterations:
    for j in range(10):
        for train_set_size_ctr in range(2, 6):
            hpc.reset_hpc_module()

            tar_patts = []
            for p in training_patterns_associative[:5*train_set_size_ctr]:
                tar_patts.append(p[1])

            print "Starting experiment 4.1, HPC chaotic recall i iters and HPC pseudopatterns..."
            hippocampal_chaotic_recall_patterns, random_ins = Experiments_4_x.experiment_4_x_1(
                    hpc, train_set_size_ctr, training_patterns_associative[:5 * train_set_size_ctr])

            Tools.save_experiment_4_1_results(hpc, random_ins, hippocampal_chaotic_recall_patterns, tar_patts,
                                              "async bug fixed", train_set_size_ctr)

            # For now, this is the ONLY place where the counter is incremented.
            Tools.increment_experiment_counter()
