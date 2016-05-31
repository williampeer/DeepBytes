import matplotlib.pyplot as plt
import numpy as np
# from scipy import stats
import Parser
import BaselineAvgs

def plot_pattern_stats_from_parsed_data_v1(parsed_data, exp_num):
    _, distinct_patterns_data = Parser.get_convergence_and_distinct_patterns_from_log_v1(parsed_data)
    perfect_recall_rates, spurious_patterns_extracted = Parser. \
        get_perfect_recall_rates_and_spurious_patterns_from_data(parsed_data)

    dist_patterns_data_experiments = []
    pf_buckets = []
    spurious_buckets = []
    for i in range(4):
        dist_patterns_data_experiments.append([])
        pf_buckets.append([])
        spurious_buckets.append([])

    for i in range(4):
        dist_patterns_data_experiments[i].append(distinct_patterns_data[0][i*100: i*100 + 100])
        dist_patterns_data_experiments[i].append(distinct_patterns_data[1][i*100: i*100 + 100])
        dist_patterns_data_experiments[i].append(distinct_patterns_data[2][i*100: i*100 + 100])
        dist_patterns_data_experiments[i].append(distinct_patterns_data[3][i*100: i*100 + 100])

        pf_buckets[i].append(perfect_recall_rates[0][i*20: i*20 + 20])
        pf_buckets[i].append(perfect_recall_rates[1][i*20: i*20 + 20])
        pf_buckets[i].append(perfect_recall_rates[2][i*20: i*20 + 20])
        pf_buckets[i].append(perfect_recall_rates[3][i*20: i*20 + 20])

        spurious_buckets[i].append(spurious_patterns_extracted[0][i*20: i*20 + 20])
        spurious_buckets[i].append(spurious_patterns_extracted[1][i*20: i*20 + 20])
        spurious_buckets[i].append(spurious_patterns_extracted[2][i*20: i*20 + 20])
        spurious_buckets[i].append(spurious_patterns_extracted[3][i*20: i*20 + 20])

    # convergence_data[0]: convergence data for the first set size.
    avg_recall, stds_avg_recall = Parser.get_average_recall_ratios(dist_patterns_data_experiments[exp_num])

    for i in range(len(spurious_buckets[exp_num])):
        for j in range(len(spurious_buckets[exp_num][i])):
            spurious_buckets[exp_num][i][j] = spurious_buckets[exp_num][i][j] / 5.

    # print spurious_buckets[0][1]
    avg_spurious_ratios, stds_spurious = Parser.get_average_recall_ratios(spurious_buckets[exp_num])

    # print avg_recall
    # print "stds_avg_recall:", stds_avg_recall
    # print
    # print avg_spurious_ratios
    # print "stds_spurious:", stds_spurious

    # graph 1: box plot
    width = .35
    V = np.arange(4)

    p1 = plt.bar(V, avg_recall, width=width, color='y', yerr=stds_avg_recall)
    p2 = plt.bar(V + width, avg_spurious_ratios, width=width, color='r', yerr=stds_spurious)

    plt.rcParams.update({'font.size': 25})
    plt.ylabel('Number of patterns')
    plt.xlabel('Set size')
    plt.title('Average number of patterns recalled for each sub-set')
    plt.xticks(V + width, ('2', '3', '4', '5'))
    plt.yticks(np.arange(0, 6, 1))
    plt.legend((p1[0], p2[0]), ('avg. # patterns recalled per sub-set', 'avg. spurious recall per sub-set'),
               bbox_to_anchor=(0.42, 1))

    plt.grid(True)
    plt.show()


def plot_convergence_ratios_for_data(parsed_data, data_points_per_experiment_config):
    convergence_data, _ = Parser.get_convergence_and_distinct_patterns_from_log_v1(parsed_data)
    # for all data: 4 buckets, containing 4x100 data points.

    # convergence_stats = Parser.get_convergence_avgs(convergence_data)
    ds = []
    for i in range(4):
        ds.append([convergence_data[0]
                   [i * data_points_per_experiment_config:i * data_points_per_experiment_config +
                                                          data_points_per_experiment_config],
                   convergence_data[1][i * data_points_per_experiment_config:i * data_points_per_experiment_config +
                                                                             data_points_per_experiment_config],
                   convergence_data[2][i * data_points_per_experiment_config:i * data_points_per_experiment_config +
                                                                             data_points_per_experiment_config],
                   convergence_data[3][i * data_points_per_experiment_config:i * data_points_per_experiment_config +
                                                                             data_points_per_experiment_config]])
    convergence_stats_async_true_mode_0 = Parser.get_convergence_avgs(ds[0])[1]
    convergence_stats_async_true_mode_1 = Parser.get_convergence_avgs(ds[1])[1]
    convergence_stats_async_false_mode_0 = Parser.get_convergence_avgs(ds[2])[1]
    convergence_stats_async_false_mode_1 = Parser.get_convergence_avgs(ds[3])[1]

    # Plotting:
    x = np.asarray([2, 3, 4, 5])
    plt.rcParams.update({'font.size': 25})

    print "y:", convergence_stats_async_true_mode_0
    # print "e:", e1.shape
    # print "e:", e1

    p1 = plt.plot(x, convergence_stats_async_true_mode_0, color='y', marker='o', linestyle='--', linewidth=3.0)
    p2 = plt.plot(x, convergence_stats_async_true_mode_1, color='g', marker='s', linestyle='--', linewidth=3.0)
    p3 = plt.plot(x, convergence_stats_async_false_mode_0, color='b', marker='^', linestyle='--', linewidth=3.0)
    p4 = plt.plot(x, convergence_stats_async_false_mode_1, color='c', marker='o', linestyle='--', linewidth=3.0)

    plt.ylabel('Average convergence rate (%)')
    plt.xlabel('Set size')
    plt.title('Average convergence rate by set size')
    plt.xticks(x, ('2x5', '3x5', '4x5', '5x5'))

    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Async. tm=0', 'Async. tm=1',
                                              'Sync. tm=0', 'Sync. tm=1'),
               bbox_to_anchor=(0.27, 0.28))
    plt.margins(0.025)
    plt.grid(True)
    plt.show()


def plot_convergence_iterations_for_data(parsed_data, exp_num):
    convergence_data, _ = Parser.get_convergence_and_distinct_patterns_from_log_v1(parsed_data)
    # for all data: 4 buckets, containing 4x100 data points.

    # convergence_stats = Parser.get_convergence_avgs(convergence_data)
    ds = []
    for i in range(4):
        ds.append([convergence_data[0][i*100:i*100 + 100], convergence_data[1][i*100:i*100 + 100],
                   convergence_data[2][i*100:i*100 + 100], convergence_data[3][i*100:i*100 + 100]])
    average_iters_convergence, sigmas = Parser.get_average_recall_ratios(ds[exp_num])

    # Plotting:
    x = np.asarray([2, 3, 4, 5])
    plt.rcParams.update({'font.size': 25})

    p1 = plt.plot(x, average_iters_convergence, color='b', marker='o', linestyle='--', linewidth=3.0)
    p2 = plt.plot(x, sigmas, color='r', marker='^', linestyle='--', linewidth=3.0)
    # p1 = plt.plot(x, [2.78, 19.02, 15.85, 15.58], color='r', marker='o', linestyle='--')

    plt.ylabel('Average #iterations before convergence')
    plt.xlabel('Set size')
    plt.title('Average #iterations before convergence by set size')
    plt.xticks(x, ('2', '3', '4', '5'))

    plt.legend((p1[0], p2[0]), ('Sync., turnover for every learnt set', 'Standard deviation'))
               # bbox_to_anchor=(0.445, 1))
    plt.grid(True)
    plt.show()


def plot_perfect_recall_rates_for_data(parsed_data, dps_per_exp_config_per_subset):
    perfect_recall_rates, spurious_patterns_data = Parser. \
        get_perfect_recall_rates_and_spurious_patterns_from_data(parsed_data)
    # print "len perfect_recall_rates[0]:", len(perfect_recall_rates[0])
    # print "perfect_recall_rates:", perfect_recall_rates

    # for all data: 4 buckets, containing 4x100 data points.

    # convergence_stats = Parser.get_convergence_avgs(convergence_data)
    ds = []
    lf = dps_per_exp_config_per_subset
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
    plt.rcParams.update({'font.size': 25})

    print "y:", perf_recall_stats_async_true_mode_0
    # print "e:", e1.shape
    # print "e:", e1

    p1 = plt.errorbar(x, perf_recall_stats_async_true_mode_0, stds_t_0, color='y', marker='o', linestyle='--')
    p2 = plt.errorbar(x, perf_recall_stats_async_true_mode_1, stds_t_1, color='g', marker='s', linestyle='--')
    p3 = plt.errorbar(x, perf_recall_stats_async_false_mode_0, stds_f_0, color='b', marker='^', linestyle='--')
    p4 = plt.errorbar(x, perf_recall_stats_async_false_mode_1, stds_f_1, color='c', marker='o', linestyle='--')

    plt.ylabel('Perfect recall rate (%)')
    plt.xlabel('Set size')
    plt.title('Average perfect recall rates by set size')
    plt.xticks(x, ('2x5', '3x5', '4x5', '5x5'))
    plt.margins(0.05)

    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Async. tm=0', 'Async. tm=1',
                                              'Sync. tm=0', 'Sync. tm=1'))
               # bbox_to_anchor=(0.377, 0.23))
    plt.grid(True)
    plt.show()


def plot_avg_perfect_extraction_and_spurious_patterns(parsed_data, dps_per_exp_config_per_subset):
    perfect_recall_rates, spurious_patterns_data = Parser. \
        get_perfect_recall_rates_and_spurious_patterns_from_data(parsed_data)
    print spurious_patterns_data
    print len(spurious_patterns_data[0])

    ds = []
    spds = []
    lf = dps_per_exp_config_per_subset
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
        perf_recall_stats_async_true_mode_0[i] = perf_recall_stats_async_true_mode_0[i] * float(2 + i)
        perf_recall_stats_async_true_mode_1[i] = perf_recall_stats_async_true_mode_1[i] * float(2 + i)
        perf_recall_stats_async_false_mode_0[i] = perf_recall_stats_async_false_mode_0[i] * float(2 + i)
        perf_recall_stats_async_false_mode_1[i] = perf_recall_stats_async_false_mode_1[i] * float(2 + i)

        stds_t_0[i] = stds_t_0[i] * float(2 + i)
        stds_t_1[i] = stds_t_1[i] * float(2 + i)
        stds_f_0[i] = stds_f_0[i] * float(2 + i)
        stds_f_1[i] = stds_f_1[i] * float(2 + i)

        spurpt0[i] = spurpt0[i] / float(2+i)
        spurpt1[i] = spurpt1[i] / float(2+i)
        spurpf0[i] = spurpf0[i] / float(2+i)
        spurpf1[i] = spurpf1[i] / float(2+i)

    # Plotting:
    width = .35
    x = np.asarray([2, 3, 4, 5])
    plt.rcParams.update({'font.size': 25})

    p1 = plt.bar(x, perf_recall_stats_async_true_mode_0, color='y', width=width/2)
    p1_e = plt.bar(x, spurpt0, color='r', width=width/2, bottom=perf_recall_stats_async_true_mode_0)
    p2 = plt.bar(x+width/2, perf_recall_stats_async_true_mode_1, color='g', width=width/2)
    p2_2 = plt.bar(x+width/2, spurpt1, color='r', width=width/2, bottom=perf_recall_stats_async_true_mode_1)
    p3 = plt.bar(x+width, perf_recall_stats_async_false_mode_0, color='b', width=width/2)
    p3_e = plt.bar(x+width, spurpf0, color='r', width=width/2, bottom=perf_recall_stats_async_false_mode_0)
    p4 = plt.bar(x+3*width/2, perf_recall_stats_async_false_mode_1, color='c', width=width/2)
    p4_e = plt.bar(x+3*width/2, spurpf1, color='r', width=width/2, bottom=perf_recall_stats_async_false_mode_1)

    plt.ylabel('Average perfectly recalled patterns per subset')
    plt.xlabel('Subset size')
    plt.title('Average perfect recall per subset')
    plt.xticks(x + width, ('2', '3', '4', '5'))
    plt.yticks(np.arange(0, 6, 1))

    plt.legend((p1[0], p2[0], p3[0], p4[0], p4_e[0]), ('Async., turnover for new sets', 'Async., turnover every iteration',
                                              'Sync., turnover for new sets', 'Sync., turnover every iteration',
                                              'Non-perfectly recalled patterns'))
               # bbox_to_anchor=(0.377, 1))
    plt.grid(True)
    plt.show()


# ================================== STRUCTURED BY TURNOVER RATE OR DG-WEIGHTING ======================================
def plot_convergence_stats_for_turnover_rates(parsed_data, log_filename):
    set_size_buckets = Parser.get_dictionary_list_of_convergence_and_perfect_recall_for_turnover_rates(
        Parser.get_data_with_turnover_rates(parsed_data, log_filename))

    # x, y_iters, std_iters, y_ratios, std_ratios
    x = [x * 0.02 for x in range(30)]
    results_2 = Parser.get_avg_convergence_for_x_and_set_size(2, set_size_buckets, x)
    results_3 = Parser.get_avg_convergence_for_x_and_set_size(3, set_size_buckets, x)
    results_4 = Parser.get_avg_convergence_for_x_and_set_size(4, set_size_buckets, x)
    results_5 = Parser.get_avg_convergence_for_x_and_set_size(5, set_size_buckets, x)

    plt.rcParams.update({'font.size': 25})
    plt.ylabel('Convergence ratio')
    plt.xlabel('Turnover rate')
    plt.title('Average convergence rate by turnover rate')

    # p2 = plt.errorbar(results_2[0], results_2[3], results_2[4])
    # p3 = plt.errorbar(results_3[0], results_3[3], results_3[4])
    # p4 = plt.errorbar(results_4[0], results_4[3], results_4[4])
    # p5 = plt.errorbar(results_5[0], results_5[3], results_5[4])

    p2 = plt.plot(results_2[0], results_2[3], linewidth=3.0)
    p3 = plt.plot(results_3[0], results_3[3], linewidth=3.0)
    p4 = plt.plot(results_4[0], results_4[3], linewidth=3.0)
    p5 = plt.plot(results_5[0], results_5[3], linewidth=3.0)

    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.58])
    plt.legend((p2[0], p3[0], p4[0], p5[0]), ('2x5', '3x5', '4x5', '5x5'),
               bbox_to_anchor=(1, 1.0155), ncol=4, fancybox=True, shadow=True)
    plt.margins(0.09)
    plt.grid(True)

    plt.show()


def plot_perfect_recall_rates_for_turnover_rates(parsed_data, log_filename):
    set_size_buckets = Parser.get_dictionary_list_of_convergence_and_perfect_recall_for_turnover_rates(
        Parser.get_data_with_turnover_rates(parsed_data, log_filename))

    # x, y_iters, std_iters, y_ratios, std_ratios
    x = [x * 0.02 for x in range(30)]
    results_2 = Parser.get_avg_perfect_recall_for_x_and_set_size(2, set_size_buckets, x)
    results_3 = Parser.get_avg_perfect_recall_for_x_and_set_size(3, set_size_buckets, x)
    results_4 = Parser.get_avg_perfect_recall_for_x_and_set_size(4, set_size_buckets, x)
    results_5 = Parser.get_avg_perfect_recall_for_x_and_set_size(5, set_size_buckets, x)

    plt.rcParams.update({'font.size': 25})
    plt.ylabel('Perfect recall rate')
    plt.xlabel('Turnover rate')
    plt.title('Average perfect recall rate by turnover rate')

    # p2 = plt.errorbar(results_2[0], results_2[1], results_2[2])
    # p3 = plt.errorbar(results_3[0], results_3[1], results_3[2])
    # p4 = plt.errorbar(results_4[0], results_4[1], results_4[2])
    # p5 = plt.errorbar(results_5[0], results_5[1], results_5[2])

    p2 = plt.plot(results_2[0], results_2[1], linewidth=3.0)
    p3 = plt.plot(results_3[0], results_3[1], linewidth=3.0)
    p4 = plt.plot(results_4[0], results_4[1], linewidth=3.0)
    p5 = plt.plot(results_5[0], results_5[1], linewidth=3.0)

    plt.legend((p2[0], p3[0], p4[0], p5[0]), ('2x5', '3x5', '4x5', '5x5'), bbox_to_anchor=(1, 1.0155), ncol=4, fancybox=True, shadow=True)
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.58])
    plt.margins(0.09)
    plt.grid(True)

    plt.show()


def plot_aggregate_figure_for_turnover_rates(parsed_data, log_file):
    set_size_buckets = Parser.get_dictionary_list_of_convergence_and_perfect_recall_for_turnover_rates(
        Parser.get_data_with_turnover_rates(parsed_data, log_file))

    plt.figure(1)
    plt.subplot(211)

    # x, y_iters, std_iters, y_ratios, std_ratios
    x = [x * 0.02 for x in range(30)]
    results_2 = Parser.get_avg_convergence_for_x_and_set_size(2, set_size_buckets, x)
    results_3 = Parser.get_avg_convergence_for_x_and_set_size(3, set_size_buckets, x)
    results_4 = Parser.get_avg_convergence_for_x_and_set_size(4, set_size_buckets, x)
    results_5 = Parser.get_avg_convergence_for_x_and_set_size(5, set_size_buckets, x)

    plt.rcParams.update({'font.size': 25})
    plt.ylabel('Convergence ratio')
    # plt.xlabel('Turnover rate')
    plt.title('Average convergence and perfect recall rates by turnover rate')

    p2 = plt.plot(results_2[0], results_2[3], linewidth=3.0)
    p3 = plt.plot(results_3[0], results_3[3], linewidth=3.0)
    p4 = plt.plot(results_4[0], results_4[3], linewidth=3.0)
    p5 = plt.plot(results_5[0], results_5[3], linewidth=3.0)

    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.58])
    plt.legend((p2[0], p3[0], p4[0], p5[0]), ('2x5', '3x5', '4x5', '5x5'),
               bbox_to_anchor=(1.115, 1.0155), ncol=1, fancybox=True, shadow=True)
    plt.margins(0.02)
    plt.grid(True)

    # x, y_iters, std_iters, y_ratios, std_ratios
    x = [x * 0.02 for x in range(30)]
    results_2 = Parser.get_avg_perfect_recall_for_x_and_set_size(2, set_size_buckets, x)
    results_3 = Parser.get_avg_perfect_recall_for_x_and_set_size(3, set_size_buckets, x)
    results_4 = Parser.get_avg_perfect_recall_for_x_and_set_size(4, set_size_buckets, x)
    results_5 = Parser.get_avg_perfect_recall_for_x_and_set_size(5, set_size_buckets, x)

    plt.subplot(212)
    # plt.rcParams.update({'font.size': 25})
    plt.ylabel('Perfect recall rate')
    plt.xlabel('Turnover rate')
    # plt.title('Average perfect recall rate by turnover rate')

    p2 = plt.plot(results_2[0], results_2[1], linewidth=3.0)
    p3 = plt.plot(results_3[0], results_3[1], linewidth=3.0)
    p4 = plt.plot(results_4[0], results_4[1], linewidth=3.0)
    p5 = plt.plot(results_5[0], results_5[1], linewidth=3.0)

    plt.legend((p2[0], p3[0], p4[0], p5[0]), ('2x5', '3x5', '4x5', '5x5'), ncol=1, bbox_to_anchor=(1.115, 1.0155),
               fancybox=True, shadow=True)
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.58])
    plt.margins(0.02)
    plt.subplots_adjust(left=0.06)
    plt.grid(True)

    plt.show()


def plot_convergence_stats_for_dg_weightings(parsed_data, additional_plot_title):
    set_size_buckets = Parser.get_dictionary_list_of_convergence_and_perfect_recall_for_dg_weightings(parsed_data)

    # x, y_iters, std_iters, y_ratios, std_ratios
    x = range(30)
    results_2 = Parser.get_avg_convergence_for_x_and_set_size(2, set_size_buckets, x)
    results_3 = Parser.get_avg_convergence_for_x_and_set_size(3, set_size_buckets, x)
    results_4 = Parser.get_avg_convergence_for_x_and_set_size(4, set_size_buckets, x)
    results_5 = Parser.get_avg_convergence_for_x_and_set_size(5, set_size_buckets, x)

    plt.rcParams.update({'font.size': 25})
    plt.ylabel('Convergence ratio')
    plt.xlabel('DG-weighting')
    plt.title('Average convergence rate by DG-weighting, ' + additional_plot_title)

    p2 = plt.errorbar(results_2[0], results_2[3], results_2[4])
    p3 = plt.errorbar(results_3[0], results_3[3], results_3[4])
    p4 = plt.errorbar(results_4[0], results_4[3], results_4[4])
    p5 = plt.errorbar(results_5[0], results_5[3], results_5[4])

    plt.legend((p2[0], p3[0], p4[0], p5[0]), ('2x5', '3x5', '4x5', '5x5'))
    plt.grid(True)
    plt.margins(0.01)

    plt.yticks(np.arange(0, 1.1, .1))

    plt.show()


def plot_convergence_stats_for_dg_weightings_no_err_bars(parsed_data, additional_plot_title):
    set_size_buckets = Parser.get_dictionary_list_of_convergence_and_perfect_recall_for_dg_weightings(parsed_data)

    # x, y_iters, std_iters, y_ratios, std_ratios
    x = range(30)
    results_2 = Parser.get_avg_convergence_for_x_and_set_size(2, set_size_buckets, x)
    results_3 = Parser.get_avg_convergence_for_x_and_set_size(3, set_size_buckets, x)
    results_4 = Parser.get_avg_convergence_for_x_and_set_size(4, set_size_buckets, x)
    results_5 = Parser.get_avg_convergence_for_x_and_set_size(5, set_size_buckets, x)

    plt.rcParams.update({'font.size': 25})
    plt.ylabel('Convergence ratio')
    plt.xlabel('DG-weighting')
    plt.title('Average convergence rate by DG-weighting, ' + additional_plot_title)

    p2 = plt.plot(results_2[0], results_2[3])
    p3 = plt.plot(results_3[0], results_3[3])
    p4 = plt.plot(results_4[0], results_4[3])
    p5 = plt.plot(results_5[0], results_5[3])

    plt.legend((p2[0], p3[0], p4[0], p5[0]), ('2x5', '3x5', '4x5', '5x5'),
               bbox_to_anchor=(1, 0.8), ncol=1, fancybox=True, shadow=True)
    plt.grid(True)
    plt.margins(0.01)

    plt.yticks(np.arange(0, 1.1, .1))

    plt.show()


def plot_perfect_recall_rates_for_dg_weightings_no_err_bars(parsed_data, additional_plot_title):
    set_size_buckets = Parser.get_dictionary_list_of_convergence_and_perfect_recall_for_dg_weightings(parsed_data)

    # x, y_iters, std_iters, y_ratios, std_ratios
    x = range(30)
    results_2 = Parser.get_avg_convergence_for_x_and_set_size(2, set_size_buckets, x)
    results_3 = Parser.get_avg_convergence_for_x_and_set_size(3, set_size_buckets, x)
    results_4 = Parser.get_avg_convergence_for_x_and_set_size(4, set_size_buckets, x)
    results_5 = Parser.get_avg_convergence_for_x_and_set_size(5, set_size_buckets, x)

    plt.rcParams.update({'font.size': 25})
    plt.ylabel('Convergence ratio')
    plt.xlabel('Turnover rate')
    plt.title('Average convergence rate by DG-weighting, ' + additional_plot_title)

    p2 = plt.plot(results_2[0], results_2[3])
    p3 = plt.plot(results_3[0], results_3[3])
    p4 = plt.plot(results_4[0], results_4[3])
    p5 = plt.plot(results_5[0], results_5[3])

    plt.legend((p2[0], p3[0], p4[0], p5[0]), ('2x5', '3x5', '4x5', '5x5'))
               # bbox_to_anchor=(1, 0.9), ncol=1, fancybox=True, shadow=True)
    plt.grid(True)
    plt.margins(0.01)

    plt.yticks(np.arange(0, 1.1, .1))

    plt.show()


def plot_aggregate_figure_for_dg_weightings(parsed_data):
    set_size_buckets = Parser.get_dictionary_list_of_convergence_and_perfect_recall_for_dg_weightings(parsed_data)

    plt.figure(1)
    plt.subplot(211)

    # x, y_iters, std_iters, y_ratios, std_ratios
    x = range(30)
    results_2 = Parser.get_avg_convergence_for_x_and_set_size(2, set_size_buckets, x)
    results_3 = Parser.get_avg_convergence_for_x_and_set_size(3, set_size_buckets, x)
    results_4 = Parser.get_avg_convergence_for_x_and_set_size(4, set_size_buckets, x)
    results_5 = Parser.get_avg_convergence_for_x_and_set_size(5, set_size_buckets, x)

    plt.rcParams.update({'font.size': 25})
    plt.ylabel('Convergence ratio')
    # plt.xlabel('DG-weighting')
    plt.title('Average convergence and perfect recall rates by DG-weighting')

    p2 = plt.plot(results_2[0], results_2[3], linewidth=3.0)
    p3 = plt.plot(results_3[0], results_3[3], linewidth=3.0)
    p4 = plt.plot(results_4[0], results_4[3], linewidth=3.0)
    p5 = plt.plot(results_5[0], results_5[3], linewidth=3.0)

    plt.xticks(np.arange(0, 30, 5))
    plt.legend((p2[0], p3[0], p4[0], p5[0]), ('2x5', '3x5', '4x5', '5x5'),
               bbox_to_anchor=(1.115, 1.0155), ncol=1, fancybox=True, shadow=True)
    plt.margins(0.02)
    plt.grid(True)

    # x, y_iters, std_iters, y_ratios, std_ratios
    results_2 = Parser.get_avg_perfect_recall_for_x_and_set_size(2, set_size_buckets, x)
    results_3 = Parser.get_avg_perfect_recall_for_x_and_set_size(3, set_size_buckets, x)
    results_4 = Parser.get_avg_perfect_recall_for_x_and_set_size(4, set_size_buckets, x)
    results_5 = Parser.get_avg_perfect_recall_for_x_and_set_size(5, set_size_buckets, x)

    plt.subplot(212)
    plt.ylabel('Perfect recall rate')
    plt.xlabel('DG-weighting')
    # plt.title('Average perfect recall rate by turnover rate')

    p2 = plt.plot(results_2[0], results_2[1], linewidth=3.0)
    p3 = plt.plot(results_3[0], results_3[1], linewidth=3.0)
    p4 = plt.plot(results_4[0], results_4[1], linewidth=3.0)
    p5 = plt.plot(results_5[0], results_5[1], linewidth=3.0)

    plt.legend((p2[0], p3[0], p4[0], p5[0]), ('2x5', '3x5', '4x5', '5x5'), ncol=1, bbox_to_anchor=(1.115, 1.0155),
               fancybox=True, shadow=True)
    plt.xticks(np.arange(0, 30, 5))
    plt.margins(0.02)
    plt.subplots_adjust(left=0.06)
    plt.grid(True)

    plt.show()


# format: [set_size, #sets, [convergence_num, distinct_patterns_recalled]
# 10 trials for the current log
# log_filename = 'Logs/DEBUGGED-DGWs.txt'
# lf1 = 'Logs/1.txt'
# lf2 = 'Logs/2.txt'
# lf3 = 'Logs/3.txt'
# lf4 = 'Logs/4.txt'
# lf5 = 'Logs/5.txt'
# lf6 = 'Logs/6.txt'
# outer_scope_parsed_data = Parser.get_data_from_log_file(log_filename)
# parsed_data1 = Parser.get_data_from_log_file(lf1)
# parsed_data2 = Parser.get_data_from_log_file(lf2)
# parsed_data3 = Parser.get_data_from_log_file(lf3)
# parsed_data4 = Parser.get_data_from_log_file(lf4)
# parsed_data5 = Parser.get_data_from_log_file(lf5)
# parsed_data6 = Parser.get_data_from_log_file(lf6)
# AggregateFigurePlots.plot_aggregate_figure_for_dg_weightings(parsed_data[1200:2400])
# AggregateFigurePlots.plot_aggregate_figure_for_turnover_rates(parsed_data1, lf1)
# AggregateFigurePlots.plot_aggregate_figure_for_turnover_rates(parsed_data2, lf2)
# AggregateFigurePlots.plot_aggregate_figure_for_turnover_rates(parsed_data3, lf3)
# AggregateFigurePlots.plot_aggregate_figure_for_turnover_rates(parsed_data4, lf4)
# AggregateFigurePlots.plot_aggregate_figure_for_turnover_rates(parsed_data5, lf5)
# AggregateFigurePlots.plot_aggregate_figure_for_turnover_rates(parsed_data6, lf6)

# plot_pattern_stats_from_parsed_data_v1(outer_scope_parsed_data, 3)
# plot_convergence_ratios_for_data(outer_scope_parsed_data, 80)
# plot_convergence_iterations_for_data(outer_scope_parsed_data, 3)
# plot_perfect_recall_rates_for_data(outer_scope_parsed_data, dps_per_exp_config_per_subset=20)
# plot_avg_perfect_extraction_and_spurious_patterns(outer_scope_parsed_data, dps_per_exp_config_per_subset=20)
# plot_convergence_stats_for_turnover_rates(outer_scope_parsed_data, log_filename)
# plot_perfect_recall_rates_for_turnover_rates(outer_scope_parsed_data, log_filename)
# plot_aggregate_figure_for_turnover_rates(outer_scope_parsed_data, log_filename)
# plot_aggregate_figure_for_dg_weightings(outer_scope_parsed_data[3600:])

# specific_plot_title = 'ASYNC., turnover rate = 0.04, turnover mode 1'
# current_data = outer_scope_parsed_data[:1200]
# plot_convergence_stats_for_dg_weightings(current_data, specific_plot_title)
# plot_convergence_stats_for_dg_weightings_no_err_bars(current_data, specific_plot_title)

lf_path_sync_15 = 'Consolidation-logs/consolidation-log sync tr30 tm1 15 iters local span output local demo ' \
                  'catastrophic forgetting reduced according to goodness of fit.txt'
lf_path_sync_50 = 'Consolidation-logs/consolidation subsets span io, sync 50 iters.txt'
lf_path_async_tm0_15 = 'Consolidation-logs/async tm0 tr50 dgw25 local, output span both local, reduced catastrophic ' \
                   'forgetting.txt'
lf_path_async_tm1_15 = 'Consolidation-logs/all-consolidation-original-schemes-async-tm1-tr30-dgw25.txt'


def plot_avgs_for_consolidation_log_all_original_consolidation_schemes(lf_path):

    exp_avgs_15 = []
    exp_avgs_200 = []
    for i in range(8):
        log_lines = Parser.retrieve_log_lines_for_experiment(lf_path, lines_per_exp=3, start_exp=i*80, num=(i*80+80))
        print log_lines
        data1, data2 = Parser.parse_data_from_neocortical_consolidation_log_lines(log_lines)
        exp_avgs_15.append(Parser.get_avgs_from_set_size_lists(data1))
        exp_avgs_200.append(Parser.get_avgs_from_set_size_lists(data2))

    # plot_from_avgs(exp_avgs_15[experiments_start: experiments_stop])
    # plot_from_avgs(exp_avgs_200[experiments_start: experiments_stop])
    # plot_from_quad_avgs([exp_avgs_15[0], exp_avgs_15[2], exp_avgs_15[4], exp_avgs_15[6]])
    # plot_from_quad_avgs([exp_avgs_200[0], exp_avgs_200[2], exp_avgs_200[4], exp_avgs_200[6]])
    # plot_from_quad_avgs([exp_avgs_15[1], exp_avgs_15[3], exp_avgs_15[5], exp_avgs_15[7]])
    plot_from_quad_avgs([exp_avgs_200[1], exp_avgs_200[3], exp_avgs_200[5], exp_avgs_200[7]])


def plot_from_quad_avgs(avgs_for_experiments):
    # print "avgs:", avgs_for_experiments
    plt.rcParams.update({'font.size': 25})
    plt.ylabel('Average goodness of fit')
    # plt.xlabel('DG-weighting')
    plt.title('Average goodness of fit by set size and consolidation scheme')

    x = [2, 3, 4, 5]
    p2 = plt.plot(x, avgs_for_experiments[0], marker='o', linestyle='--', linewidth=3.0)
    p3 = plt.plot(x, avgs_for_experiments[1], marker='o', linestyle='--', linewidth=3.0)
    p4 = plt.plot(x, avgs_for_experiments[2], marker='o', linestyle='--', linewidth=3.0)
    p5 = plt.plot(x, avgs_for_experiments[3], marker='o', linestyle='--', linewidth=3.0)

    plt.xticks(range(2,6), ['2x5', '3x5', '4x5', '5x5'])
    plt.xlabel('Set size')
    plt.legend((p2[0], p3[0], p4[0], p5[0]),
               ('Chaotic P\'s', '+ P. I & II', 'only + P. I', 'only + P. II'),
               bbox_to_anchor=(1, 1.0), ncol=4, fancybox=True, shadow=True)
    plt.margins(0.14)
    plt.grid(True)
    plt.show()


def plot_from_avgs(avgs_for_experiments):
    # print "avgs:", avgs_for_experiments
    plt.rcParams.update({'font.size': 25})
    plt.ylabel('Average goodness of fit')
    # plt.xlabel('DG-weighting')
    plt.title('Average goodness of fit by set size and consolidation scheme')

    x = [2, 3, 4, 5]
    p2 = plt.plot(x, avgs_for_experiments[0], marker='o', linestyle='--', linewidth=3.0)
    p6 = plt.plot(x, avgs_for_experiments[4], marker='o', linestyle='--', linewidth=3.0)

    p3 = plt.plot(x, avgs_for_experiments[1], marker='o', linestyle='--', linewidth=3.0)
    p7 = plt.plot(x, avgs_for_experiments[5], marker='o', linestyle='--', linewidth=3.0)

    p4 = plt.plot(x, avgs_for_experiments[2], marker='o', linestyle='--', linewidth=3.0)
    p9 = plt.plot(x, avgs_for_experiments[6], marker='o', linestyle='--', linewidth=3.0)

    p5 = plt.plot(x, avgs_for_experiments[3], marker='o', linestyle='--', linewidth=3.0)
    p10 = plt.plot(x, avgs_for_experiments[7], marker='o', linestyle='--', linewidth=3.0)

    p8 = plt.plot(x, BaselineAvgs.avgs, marker='^', linestyle='-', color='black', linewidth=3.0)

    plt.xticks(range(2,6), ['2x5', '3x5', '4x5', '5x5'])
    plt.xlabel('Set size')
    plt.legend((p2[0], p6[0], p3[0], p7[0], p4[0], p9[0], p5[0], p10[0], p8[0]),
               ('(15) Sync 15 iters', '(200) Sync 15 iters', '(15) Sync 50 iters', '(200) Sync 50 iters',
                '(15) Async 15 iters, tm=0', '(200) Async 15 iters, tm=0', '(15) Async 15 iters, tm=1',
                '(200) Async 15 iters, tm=1', '(15) Baseline averages'),
               bbox_to_anchor=(1, 1.0), ncol=1, fancybox=True, shadow=True)
    plt.margins(0.14)
    plt.grid(True)
    plt.show()


def plot_from_main_measures(avgs_for_experiments):
    # print "avgs:", avgs_for_experiments
    plt.rcParams.update({'font.size': 25})
    plt.ylabel('Average goodness of fit')
    # plt.xlabel('DG-weighting')
    plt.title('Measure by set size for model scheme')

    x = [2, 3, 4, 5]
    p2 = plt.plot(x, avgs_for_experiments[0], marker='o', linestyle='-', linewidth=3.0)
    p3 = plt.plot(x, avgs_for_experiments[1], marker='s', linestyle='-', linewidth=3.0)
    p4 = plt.plot(x, avgs_for_experiments[2], marker='D', linestyle='-', linewidth=3.0)
    p8 = plt.plot(x, BaselineAvgs.avgs, marker='^', linestyle='--', color='black', linewidth=3.0)

    plt.xticks(range(2,6), ['2x5', '3x5', '4x5', '5x5'])
    plt.xlabel('Set size')
    plt.legend((p2[0], p3[0], p4[0], p8[0]), ('Perfect recall rates', 'Spurious recall rates', '(15) Goodness of fit', '(15) Baseline averages'),
               bbox_to_anchor=(1, 1.0), ncol=2, fancybox=True, shadow=True)
    plt.margins(0.20)
    plt.grid(True)
    plt.show()

# plot_avgs_for_consolidation_log_all_original_consolidation_schemes(lf_path_async_tm1_15)
# lfs = [lf_path_sync_15, lf_path_sync_50, lf_path_async_tm0_15, lf_path_async_tm1_15]
# avgs_15 = []
# avgs_200 = []
# for lf in lfs:
#     parsed_data = Parser.parse_data_from_neocortical_consolidation_log_lines(
#         Parser.retrieve_log_lines_for_experiment(lf, 3, 0, 80))
#     avgs_15.append(Parser.get_avgs_from_set_size_lists(parsed_data[0]))
#     avgs_200.append(Parser.get_avgs_from_set_size_lists(parsed_data[1]))
# plot_from_avgs(avgs_for_experiments=avgs_15 + avgs_200)
# plot_from_avgs(avgs_for_experiments=avgs_200)

lf_cur = 'Logs/relaxed-criterion-logs/homo/sync-tm1-tr50-dgw25-local.txt'
parsed_data_async_15 = Parser.get_data_from_log_file_i_iters_schemes(lf_cur, 5)  # 1 config.
prr_by_config_list, spurious_by_config_list, stds_prr, stds_spurious = \
    Parser.get_avg_perfect_recall_and_avg_spurious_recall_from_data_for_configs(
        parsed_data_async_15, iterations_per_config=20, num_of_configs=1)

# print "prr_by_config_list:", prr_by_config_list[0]
# print "spurious_by_config_list:", spurious_by_config_list[0]
perfect_recall_avgs = prr_by_config_list[0]
spurious_avgs = spurious_by_config_list[0]
goodness_avgs = Parser.get_avgs_from_set_size_lists(
    Parser.parse_data_from_neocortical_consolidation_log_lines(
        Parser.retrieve_log_lines_for_experiment('Consolidation-logs/consolidation-log-sync-tm1-tr50-local.txt', 3, 0, 80)
    )[0]
)
plot_from_main_measures([perfect_recall_avgs, spurious_avgs, goodness_avgs])
