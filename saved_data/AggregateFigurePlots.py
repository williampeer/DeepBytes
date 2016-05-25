import Parser
import matplotlib.pyplot as plt
import numpy as np
import ThreeDBarPlot


def plot_aggregate_figure_for_dg_weightings(parsed_data):
    values_prrs, values_spurious = ThreeDBarPlot.process_3d_data(parsed_data, iterations_per_config=10,
                                                                 num_of_configs=30)
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
    plt.ylabel('Convergence rate')
    # plt.xlabel('DG-weighting')
    plt.title('Average convergence and recall rates by DG-weighting')

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
    plt.ylabel('Recall rate')
    plt.xlabel('DG-weighting')
    # plt.title('Average perfect recall rate by turnover rate')

    p2 = plt.plot(results_2[0], results_2[1], linewidth=3.0)
    p3 = plt.plot(results_3[0], results_3[1], linewidth=3.0)
    p4 = plt.plot(results_4[0], results_4[1], linewidth=3.0)
    p5 = plt.plot(results_5[0], results_5[1], linewidth=3.0)

    plt.subplot(212)
    s_x, s_y, s_z = ThreeDBarPlot.unwrap_values(values_spurious)

    s1, s2, s3, s4 = get_set_size_values_from_aggregate(s_z)
    p6 = plt.plot(x, s1, linewidth=3.0, linestyle='--', color='b')
    p7 = plt.plot(x, s2, linewidth=3.0, linestyle='--', color='g')
    p8 = plt.plot(x, s3, linewidth=3.0, linestyle='--', color='r')
    p9 = plt.plot(x, s4, linewidth=3.0, linestyle='--', color='c')

    plt.legend((p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0]),
               ('2x5', '3x5', '4x5', '5x5', '2x5', '3x5', '4x5', '5x5'), ncol=1, bbox_to_anchor=(1.115, 1.0295),
               fancybox=True, shadow=True)
    plt.xticks(np.arange(0, 30, 5))
    plt.margins(0.02)
    plt.subplots_adjust(left=0.06)
    plt.grid(True)

    plt.show()


def plot_aggregate_figure_for_turnover_rates(parsed_data, log_file):
    set_size_buckets = Parser.get_dictionary_list_of_convergence_and_perfect_recall_for_turnover_rates(
        Parser.get_data_with_turnover_rates(parsed_data, log_file))
    values_prrs, values_spurious = ThreeDBarPlot.process_3d_data(parsed_data, iterations_per_config=10,
                                                                 num_of_configs=30)

    plt.figure(1)
    plt.subplot(211)

    # x, y_iters, std_iters, y_ratios, std_ratios
    x = [x * 0.02 for x in range(30)]
    results_2 = Parser.get_avg_convergence_for_x_and_set_size(2, set_size_buckets, x)
    results_3 = Parser.get_avg_convergence_for_x_and_set_size(3, set_size_buckets, x)
    results_4 = Parser.get_avg_convergence_for_x_and_set_size(4, set_size_buckets, x)
    results_5 = Parser.get_avg_convergence_for_x_and_set_size(5, set_size_buckets, x)

    plt.rcParams.update({'font.size': 25})
    plt.ylabel('Convergence rate')
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
    plt.ylabel('Recall rate')
    plt.xlabel('Turnover rate')
    # plt.title('Average perfect recall rate by turnover rate')

    p2 = plt.plot(results_2[0], results_2[1], linewidth=3.0)
    p3 = plt.plot(results_3[0], results_3[1], linewidth=3.0)
    p4 = plt.plot(results_4[0], results_4[1], linewidth=3.0)
    p5 = plt.plot(results_5[0], results_5[1], linewidth=3.0)

    plt.subplot(212)
    s_x, s_y, s_z = ThreeDBarPlot.unwrap_values(values_spurious)

    s1, s2, s3, s4 = get_set_size_values_from_aggregate(s_z)
    p6 = plt.plot(x, s1, linewidth=3.0, linestyle='--', color='b')
    p7 = plt.plot(x, s2, linewidth=3.0, linestyle='--', color='g')
    p8 = plt.plot(x, s3, linewidth=3.0, linestyle='--', color='r')
    p9 = plt.plot(x, s4, linewidth=3.0, linestyle='--', color='c')

    plt.legend((p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0]),
               ('2x5', '3x5', '4x5', '5x5', '2x5', '3x5', '4x5', '5x5'), ncol=1, bbox_to_anchor=(1.115, 1.0295),
               fancybox=True, shadow=True)
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.58])
    plt.margins(0.02)
    plt.subplots_adjust(left=0.06)
    plt.grid(True)

    plt.show()


def get_set_size_values_from_aggregate(input_list):
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    for ctr in range(len(input_list)/4):
        s1.append(input_list[ctr*4])
        s2.append(input_list[ctr*4+1])
        s3.append(input_list[ctr*4+2])
        s4.append(input_list[ctr*4+3])
    return [s1, s2, s3, s4]
