import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import Parser

# format: [set_size, #sets, [convergence_num, distinct_patterns_recalled]
# 10 trials for the current log
parsed_data = Parser.get_data_from_log_file('log.txt')
# print parsed_data[0][3:]

convergence_data, distinct_patterns_data = Parser.get_convergence_and_distinct_patterns_from_log_v1(parsed_data)
# convergence_data[0]: convergence data for the first set size.