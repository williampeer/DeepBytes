import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import Parser

# format: [set_size, #sets, [convergence_num, distinct_patterns_recalled]
# 10 trials for the current log
parsed_data = Parser.get_data_from_log_file('log.txt')
# print parsed_data[0][3:]

convergence_data, distinct_patterns_data = Parser.get_convergence_and_distinct_patterns_from_log_v1(parsed_data)
perfect_recall_rates, spurious_patterns_extracted = Parser.\
    get_perfect_recall_rates_and_spurious_patterns_from_data(parsed_data)
# convergence_data[0]: convergence data for the first set size.

# for each scheme, plot two graphs
convergence_stats = Parser.get_convergence_avgs(convergence_data)
avg_recall_ratios, stds_avg_recall = Parser.get_average_recall_ratios(distinct_patterns_data)

for i in range(len(spurious_patterns_extracted)):
    for j in range(len(spurious_patterns_extracted[i])):
        spurious_patterns_extracted[i][j] = spurious_patterns_extracted[i][j] / 5.
avg_spurious_ratios, stds_spurious = Parser.get_average_recall_ratios(spurious_patterns_extracted)

print avg_recall_ratios
print stds_avg_recall
print avg_spurious_ratios
print stds_spurious

# graph 1: box plot
V = np.arange(4)

plt.bar(V, avg_recall_ratios, width=.5, color='y', yerr=stds_avg_recall)
p2 = plt.bar(V, avg_spurious_ratios, width=.5, color='r',
             bottom=avg_recall_ratios, yerr=stds_spurious)

plt.ylabel('Distinct recalled patterns')
plt.title('Patterns recalled by set size')
plt.xticks(V + .5/2., ('2', '3', '4', '5'))
# plt.yticks(np.arange(0, 81, 10))
# plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()
