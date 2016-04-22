import matplotlib.pyplot as plt
import numpy as np
# from scipy import stats
import Parser


def plot_pattern_stats_from_parsed_data_v1(parsed_data):
    _, distinct_patterns_data = Parser.get_convergence_and_distinct_patterns_from_log_v1(parsed_data)

    perfect_recall_rates, spurious_patterns_extracted = Parser.\
        get_perfect_recall_rates_and_spurious_patterns_from_data(parsed_data)
    # convergence_data[0]: convergence data for the first set size.
    avg_recall, stds_avg_recall = Parser.get_average_recall_ratios(distinct_patterns_data)

    for i in range(len(spurious_patterns_extracted)):
        for j in range(len(spurious_patterns_extracted[i])):
            spurious_patterns_extracted[i][j] = spurious_patterns_extracted[i][j] / 5.
    avg_spurious_ratios, stds_spurious = Parser.get_average_recall_ratios(spurious_patterns_extracted)

    print avg_recall
    # for i in range(len(avg_recall_ratios)):
    #     avg_recall_ratios[i] = avg_recall_ratios[i] - avg_spurious_ratios[i]
    #
    # print avg_recall_ratios
    # print stds_avg_recall
    # print
    # print avg_spurious_ratios
    # print stds_spurious

    # graph 1: box plot
    width = .35
    V = np.arange(4)

    p1 = plt.bar(V, avg_recall, width=width, color='y', yerr=stds_avg_recall)
    p2 = plt.bar(V + width, avg_spurious_ratios, width=width, color='r', yerr=stds_spurious)

    plt.rcParams.update({'font.size': 20})
    plt.ylabel('Number of patterns')
    plt.xlabel('Set size')
    plt.title('Average number of patterns recalled for each sub-set')
    plt.xticks(V + width, ('2', '3', '4', '5'))
    plt.yticks(np.arange(0, 6, 1))
    plt.legend((p1[0], p2[0]), ('avg. # patterns recalled per sub-set', 'avg. spurious recall per sub-set'),
               bbox_to_anchor=(0.42, 1))

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
    average_iters_convergence, sigmas = Parser.get_average_recall_ratios(ds[0])

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

    plt.legend((p1[0], p2[0]), ('Async., turnover between learnt sets', 'Standard deviation'),
               bbox_to_anchor=(0.428, 1))
    plt.show()


def plot_perfect_recall_rates_for_data(parsed_data):
    perfect_recall_rates, spurious_patterns_data = Parser. \
        get_perfect_recall_rates_and_spurious_patterns_from_data(parsed_data)
    # print "len perfect_recall_rates[0]:", len(perfect_recall_rates[0])
    # print "perfect_recall_rates:", perfect_recall_rates

    # for all data: 4 buckets, containing 4x100 data points.

    # convergence_stats = Parser.get_convergence_avgs(convergence_data)
    ds = []
    lf = 20
    for i in range(4):
        ds.append([perfect_recall_rates[0][i*lf:i*lf + lf], perfect_recall_rates[1][i*lf:i*lf + lf],
                   perfect_recall_rates[2][i*lf:i*lf + lf], perfect_recall_rates[3][i*lf:i*lf + lf]])
    perf_recall_stats_async_true_mode_0, stds_t_0 = Parser.get_average_recall_ratios(ds[0])
    perf_recall_stats_async_true_mode_1, stds_t_1 = Parser.get_average_recall_ratios(ds[1])
    perf_recall_stats_async_false_mode_0, stds_f_0 = Parser.get_average_recall_ratios(ds[2])
    perf_recall_stats_async_false_mode_1, stds_f_1 = Parser.get_average_recall_ratios(ds[3])

    for i in range(len(perf_recall_stats_async_true_mode_0)):
        perf_recall_stats_async_true_mode_0[i] = perf_recall_stats_async_true_mode_0[i] * 100
        perf_recall_stats_async_true_mode_1[i] = perf_recall_stats_async_true_mode_1[i] * 100
        perf_recall_stats_async_false_mode_0[i] = perf_recall_stats_async_false_mode_0[i] * 100
        perf_recall_stats_async_false_mode_1[i] = perf_recall_stats_async_false_mode_1[i] * 100

        stds_t_0[i] = stds_t_0[i] * 100
        stds_t_1[i] = stds_t_1[i] * 100
        stds_f_0[i] = stds_f_0[i] * 100
        stds_f_1[i] = stds_f_1[i] * 100

    # Plotting:
    x = np.asarray([2, 3, 4, 5])
    plt.rcParams.update({'font.size': 20})

    print "y:", perf_recall_stats_async_true_mode_0
    # print "e:", e1.shape
    # print "e:", e1

    p1 = plt.errorbar(x, perf_recall_stats_async_true_mode_0, stds_t_0, color='y', marker='o', linestyle='--')
    p2 = plt.errorbar(x, perf_recall_stats_async_true_mode_1, stds_t_1, color='g', marker='s', linestyle='--')
    p3 = plt.errorbar(x, perf_recall_stats_async_false_mode_0, stds_f_0, color='b', marker='^', linestyle='--')
    p4 = plt.errorbar(x, perf_recall_stats_async_false_mode_1, stds_f_1, color='r', marker='o', linestyle='--')

    plt.ylabel('Perfect recall rate (%)')
    plt.xlabel('Set size')
    plt.title('Average perfect recall rates by set size')
    plt.xticks(x, ('2', '3', '4', '5'))

    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Async., turnover for new sets', 'Async., turnover every iteration',
                                              'Sync., turnover for new sets', 'Sync., turnover every iteration'))
               # bbox_to_anchor=(0.377, 0.23))
    plt.show()


def plot_avg_perfect_extraction_and_spurious_patterns(parsed_data):
    perfect_recall_rates, spurious_patterns_data = Parser. \
        get_perfect_recall_rates_and_spurious_patterns_from_data(parsed_data)
    print spurious_patterns_data
    print len(spurious_patterns_data[0])

    ds = []
    spds = []
    lf = 20
    for i in range(4):
        ds.append([perfect_recall_rates[0][i*lf:i*lf + lf], perfect_recall_rates[1][i*lf:i*lf + lf],
                   perfect_recall_rates[2][i*lf:i*lf + lf], perfect_recall_rates[3][i*lf:i*lf + lf]])
        spds.append([spurious_patterns_data[0][i*lf:i*lf + lf], spurious_patterns_data[1][i*lf:i*lf + lf],
                     spurious_patterns_data[2][i*lf:i*lf + lf], spurious_patterns_data[3][i*lf:i*lf + lf]])
    perf_recall_stats_async_true_mode_0, stds_t_0 = Parser.get_average_recall_ratios(ds[0])
    perf_recall_stats_async_true_mode_1, stds_t_1 = Parser.get_average_recall_ratios(ds[1])
    perf_recall_stats_async_false_mode_0, stds_f_0 = Parser.get_average_recall_ratios(ds[2])
    perf_recall_stats_async_false_mode_1, stds_f_1 = Parser.get_average_recall_ratios(ds[3])

    spurpt0, stdspurpt0 = Parser.get_average_recall_ratios(spds[0])
    spurpt1, stdspurpt1 = Parser.get_average_recall_ratios(spds[1])
    spurpf0, stdspurpf0 = Parser.get_average_recall_ratios(spds[2])
    spurpf1, stdspurpf1 = Parser.get_average_recall_ratios(spds[3])

    for i in range(len(perf_recall_stats_async_true_mode_0)):
        perf_recall_stats_async_true_mode_0[i] = perf_recall_stats_async_true_mode_0[i] * (2 + i)
        perf_recall_stats_async_true_mode_1[i] = perf_recall_stats_async_true_mode_1[i] * (2 + i)
        perf_recall_stats_async_false_mode_0[i] = perf_recall_stats_async_false_mode_0[i] * (2 + i)
        perf_recall_stats_async_false_mode_1[i] = perf_recall_stats_async_false_mode_1[i] * (2 + i)

        stds_t_0[i] = stds_t_0[i] * (2 + i)
        stds_t_1[i] = stds_t_1[i] * (2 + i)
        stds_f_0[i] = stds_f_0[i] * (2 + i)
        stds_f_1[i] = stds_f_1[i] * (2 + i)

        spurpt0[i] = spurpt0[i] / float(2+i)
        spurpt1[i] = spurpt1[i] / float(2+i)
        spurpf0[i] = spurpf0[i] / float(2+i)
        spurpf1[i] = spurpf1[i] / float(2+i)

    # Plotting:
    width = .35
    x = np.asarray([2, 3, 4, 5])
    plt.rcParams.update({'font.size': 20})

    p1 = plt.bar(x, perf_recall_stats_async_true_mode_0, color='y', width=width/2)
    p1_e = plt.bar(x, spurpt0, color='r', width=width/2, bottom=perf_recall_stats_async_true_mode_0)
    p2 = plt.bar(x+width/2, perf_recall_stats_async_true_mode_1, color='g', width=width/2)
    p2_2 = plt.bar(x+width/2, spurpt1, color='r', width=width/2, bottom=perf_recall_stats_async_true_mode_1)
    p3 = plt.bar(x+width, perf_recall_stats_async_false_mode_0, color='b', width=width/2)
    p3_e = plt.bar(x+width, spurpf0, color='r', width=width/2, bottom=perf_recall_stats_async_false_mode_0)
    p4 = plt.bar(x+3*width/2, perf_recall_stats_async_false_mode_1, color='c', width=width/2)
    p4_e = plt.bar(x+3*width/2, spurpf1, color='r', width=width/2, bottom=perf_recall_stats_async_false_mode_1)

    plt.ylabel('Perfectly recalled patterns')
    plt.xlabel('Set size')
    plt.title('Average perfect recall by set size')
    plt.xticks(x + width, ('2', '3', '4', '5'))
    plt.yticks(np.arange(0, 6, 1))

    plt.legend((p1[0], p2[0], p3[0], p4[0], p1_e[0]), ('Async., turnover for new sets',
                                                       'Async., turnover every iteration',
                                                       'Sync., turnover for new sets',
                                                       'Sync., turnover every iteration',
                                                       'Spuriously recalled patterns'))
               # bbox_to_anchor=(0.377, 1))
    plt.show()


# format: [set_size, #sets, [convergence_num, distinct_patterns_recalled]
# 10 trials for the current log
outer_scope_parsed_data = Parser.get_data_from_log_file('log.txt')
# plot_pattern_stats_from_parsed_data_v1(outer_scope_parsed_data)
# plot_convergence_ratios_for_data(outer_scope_parsed_data)
plot_convergence_iterations_for_data(outer_scope_parsed_data)
# plot_perfect_recall_rates_for_data(outer_scope_parsed_data)
# plot_avg_perfect_extraction_and_spurious_patterns(outer_scope_parsed_data)
