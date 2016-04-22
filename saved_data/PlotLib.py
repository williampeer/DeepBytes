import matplotlib.pyplot as plt
import numpy as np
# from scipy import stats
import Parser


def plot_pattern_stats_from_parsed_data_v1(parsed_data):
    _, distinct_patterns_data = Parser.get_convergence_and_distinct_patterns_from_log_v1(parsed_data)

    perfect_recall_rates, spurious_patterns_extracted = Parser.\
        get_perfect_recall_rates_and_spurious_patterns_from_data(parsed_data)
    # convergence_data[0]: convergence data for the first set size.
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

    plt.rcParams.update({'font.size': 20})
    plt.ylabel('Average number of patterns recalled')
    plt.xlabel('Set size')
    plt.title('Average number of distinct patterns recalled by set size')
    plt.xticks(V + width/2., ('2', '3', '4', '5'))
    # plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0]), ('Perfectly recalled patterns', 'Non-perfect or spurious recall'))
               #bbox_to_anchor=(0.36, 1))

    plt.show()


def plot_convergence_ratios_for_data(parsed_data):
    convergence_data, _ = Parser.get_convergence_and_distinct_patterns_from_log_v1(parsed_data)
    # for all data: 4 buckets, containing 4x100 data points.

    # convergence_stats = Parser.get_convergence_avgs(convergence_data)
    ds = []
    for i in range(4):
        ds.append([convergence_data[0][i*100:i*100 + 100], convergence_data[1][i*100:i*100 + 100],
                   convergence_data[2][i*100:i*100 + 100], convergence_data[3][i*100:i*100 + 100]])
    convergence_stats_async_true_mode_0 = Parser.get_convergence_avgs(ds[0])[1]
    convergence_stats_async_true_mode_1 = Parser.get_convergence_avgs(ds[1])[1]
    convergence_stats_async_false_mode_0 = Parser.get_convergence_avgs(ds[2])[1]
    convergence_stats_async_false_mode_1 = Parser.get_convergence_avgs(ds[3])[1]

    e1 = []
    for i in range(4):
        e1.append(Parser.get_standard_deviation_from_values((np.array(ds[0]) < 50)[i]))
    e1 = np.asarray(e1)

    # Plotting:
    width = 0.35
    x = np.asarray([2, 3, 4, 5])
    plt.rcParams.update({'font.size': 20})

    print "y:", convergence_stats_async_true_mode_0
    # print "e:", e1.shape
    # print "e:", e1

    p1 = plt.plot(x, convergence_stats_async_true_mode_0, color='y', marker='o', linestyle='--')
    p2 = plt.plot(x, convergence_stats_async_true_mode_1, color='g', marker='s', linestyle='--')
    p3 = plt.plot(x, convergence_stats_async_false_mode_0, color='b', marker='^', linestyle='--')
    p4 = plt.plot(x, convergence_stats_async_false_mode_1, color='r', marker='o', linestyle='--')

    plt.ylabel('Average convergence rate (%)')
    plt.xlabel('Set size')
    plt.title('Average convergence rate by set size')
    plt.xticks(x, ('2', '3', '4', '5'))

    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Async., turnover for new sets', 'Async., turnover every iteration',
                                              'Sync., turnover for new sets', 'Sync., turnover every iteration'),
               bbox_to_anchor=(0.377, 0.23))
    plt.show()


def plot_convergence_iterations_for_data(parsed_data):
    convergence_data, _ = Parser.get_convergence_and_distinct_patterns_from_log_v1(parsed_data)
    # for all data: 4 buckets, containing 4x100 data points.

    # convergence_stats = Parser.get_convergence_avgs(convergence_data)
    ds = []
    for i in range(4):
        ds.append([convergence_data[0][i*100:i*100 + 100], convergence_data[1][i*100:i*100 + 100],
                   convergence_data[2][i*100:i*100 + 100], convergence_data[3][i*100:i*100 + 100]])
    average_iters_convergence, sigmas = Parser.get_average_recall_ratios(ds[2])

    # Plotting:
    x = np.asarray([2, 3, 4, 5])
    plt.rcParams.update({'font.size': 20})

    p1 = plt.plot(x, average_iters_convergence, color='b', marker='o', linestyle='--')
    p2 = plt.plot(x, sigmas, color='r', marker='^', linestyle='--')
    # p1 = plt.plot(x, [2.78, 19.02, 15.85, 15.58], color='r', marker='o', linestyle='--')

    plt.ylabel('Average #iterations before convergence')
    plt.xlabel('Set size')
    plt.title('Average #iterations before convergence by set size')
    plt.xticks(x, ('2', '3', '4', '5'))

    plt.legend((p1[0], p2[0]), ('Sync., turnover between learnt sets', 'Standard deviation'))
               # bbox_to_anchor=(0.377, 0.23))
    plt.show()

# format: [set_size, #sets, [convergence_num, distinct_patterns_recalled]
# 10 trials for the current log
outer_scope_parsed_data = Parser.get_data_from_log_file('log.txt')
plot_pattern_stats_from_parsed_data_v1(outer_scope_parsed_data)
# plot_convergence_ratios_for_data(outer_scope_parsed_data)
# plot_convergence_iterations_for_data(outer_scope_parsed_data)
