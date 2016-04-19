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

# for each scheme, plot two graphs
convergence_stats = Parser.get_convergence_avgs(convergence_data)
avg_recall_ratios = Parser.get_average_recall_ratios(distinct_patterns_data)

print convergence_stats
print avg_recall_ratios

# graph 1: box plot
stds = []
i = 0
for ds in distinct_patterns_data:
    stds.append(Parser.get_standard_deviation(avg_recall_ratios[i], ds))
    i += 1
V = np.arange(4)
print stds

plt.bar(V, avg_recall_ratios, width=.5, color='y', yerr=stds)

plt.ylabel('Distinct recalled patterns')
plt.title('Patterns recalled by set size')
plt.xticks(V + .5/2., ('2', '3', '4', '5'))
# plt.yticks(np.arange(0, 81, 10))
# plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()
