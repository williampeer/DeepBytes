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
for i in range(len(avg_recall_ratios)):
    avg_recall_ratios[i] = avg_recall_ratios[i] - avg_spurious_ratios[i]

print avg_recall_ratios
print stds_avg_recall
print
print avg_spurious_ratios
print stds_spurious

# graph 1: box plot
width = .35
V = np.arange(4)

p1 = plt.bar(V, avg_recall_ratios, width=width, color='y')  #, yerr=stds_avg_recall)
p2 = plt.bar(V, avg_spurious_ratios, width=width, color='r',
             bottom=avg_recall_ratios, yerr=stds_avg_recall)

plt.ylabel('Average number of patterns recalled')
plt.xlabel('Set size')
plt.title('Average number of distinct patterns recalled by set size')
plt.xticks(V + width/2., ('2', '3', '4', '5'))
# plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Perfectly recalled patterns', 'Non-perfect or spurious recall'),
           bbox_to_anchor=(0.326, 1))

plt.show()
