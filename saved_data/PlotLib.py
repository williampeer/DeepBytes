import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import Parser

# format: [set_size, #sets, [convergence_num, distinct_patterns_recalled]
# 10 trials for the current log
parsed_data = Parser.get_data_from_log_file('log.txt')
# print parsed_data[0][3:]

experiment_data_convergence = []
experiment_data_distinct_patterns = []
for i in range(5):
    experiment_data_convergence.append([])
    experiment_data_distinct_patterns.append([])

num_of_experiments = len(parsed_data)
for i in range(num_of_experiments):
    set_size = parsed_data[i][0]
    num_of_sets = parsed_data[i][1]
    dg_weighting = parsed_data[i][2]

    for data_tuple in parsed_data[i][3:]:
        iterations_to_convergence_or_max = data_tuple[0]
        distinct_patterns_recalled = data_tuple[1]

        experiment_data_convergence[set_size-2].append(iterations_to_convergence_or_max)
        experiment_data_distinct_patterns[set_size-2].append(distinct_patterns_recalled)

print "data for a number of sets =:", len(experiment_data_distinct_patterns[0])
print experiment_data_distinct_patterns[0]
print experiment_data_distinct_patterns[0][:25]
print experiment_data_distinct_patterns[0][25:50]
print experiment_data_distinct_patterns[0][50:75]
print experiment_data_distinct_patterns[0][75:100]
