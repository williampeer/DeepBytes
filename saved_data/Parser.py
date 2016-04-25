import numpy as np


# returns: [ experiment: [set_size, #sets, DGW, 5x tuples: [iters_to_convergence, distinct_patterns_recalled]]]
def get_data_from_log_file(filename):
    log_file = file(filename, 'r')

    contents = log_file.read()
    log_file.close()

    lines = contents.split('\n')
    all_data = []
    for experiment_ctr in range(len(lines)/13):
        experiment_data = []

        sep = lines[13*experiment_ctr].split('x')
        set_size = int(sep[0][len(sep[0])-1])
        training_sets = int(sep[1][0])

        words = lines[experiment_ctr * 13].split()
        dg_weighting = words[-1][:len(words[-1])-1]

        perfect_recall_rate = float(lines[13 * experiment_ctr + 11].split()[-1])
        num_of_spurious_patterns = int(lines[13 * experiment_ctr + 12].split()[-1])

        experiment_data.append(set_size)
        experiment_data.append(training_sets)
        experiment_data.append(float(dg_weighting))
        experiment_data.append(perfect_recall_rate)
        experiment_data.append(num_of_spurious_patterns)

        for i in range(5):
            convergence_line_words = lines[13 * experiment_ctr + 1 + 2 * i].split()
            patterns_recalled_line_words = lines[13 * experiment_ctr + 2 + 2 * i].split()

            iterations_before_convergence = int(convergence_line_words[2])
            distinct_patterns_recalled = int(patterns_recalled_line_words[1])
            experiment_data.append([iterations_before_convergence, distinct_patterns_recalled])

        all_data.append(experiment_data)

    return all_data


def get_convergence_and_distinct_patterns_from_log_v1(parsed_data):
    experiment_data_convergence = []
    experiment_data_distinct_patterns = []
    for i in range(4):
        experiment_data_convergence.append([])
        experiment_data_distinct_patterns.append([])

    num_of_experiments = len(parsed_data)
    for i in range(num_of_experiments):
        set_size = parsed_data[i][0]
        num_of_sets = parsed_data[i][1]
        dg_weighting = parsed_data[i][2]

        for data_tuple in parsed_data[i][5:]:
            iterations_to_convergence_or_max = data_tuple[0]
            distinct_patterns_recalled = data_tuple[1]

            experiment_data_convergence[set_size-2].append(iterations_to_convergence_or_max)
            experiment_data_distinct_patterns[set_size-2].append(distinct_patterns_recalled)

    return [experiment_data_convergence, experiment_data_distinct_patterns]


def get_perfect_recall_rates_and_spurious_patterns_from_data(parsed_data):
    perf_recall_data = []
    spurious_patts_data = []
    for i in range(4):
        perf_recall_data.append([])
        spurious_patts_data.append([])

    num_of_experiments = len(parsed_data)
    for i in range(num_of_experiments):
        set_size = parsed_data[i][0]
        perf_recall_data[set_size-2].append(parsed_data[i][3])
        spurious_patts_data[set_size-2].append(parsed_data[i][4])

    return [perf_recall_data, spurious_patts_data]


def get_convergence_info_and_distinct_patterns_from_data(set_size, convergence_data, distinct_patterns_data):
    converged_ctr = 0
    convergence_sum = 0
    for converge_el in convergence_data:
        convergence_sum += converge_el
        if converge_el < 50:
            converged_ctr += 1

    distinct_patterns_sum = 0
    for el in distinct_patterns_data:
        distinct_patterns_sum += el

    avg_convergence_iters = convergence_sum / float(len(convergence_data))
    convergence_ratio = converged_ctr / float(len(convergence_data))

    avg_dist_patterns = distinct_patterns_sum / float(len(distinct_patterns_data))
    recall_ratio = avg_dist_patterns / float(set_size)

    return [[avg_convergence_iters, convergence_ratio], [avg_dist_patterns, recall_ratio]]


def get_convergence_avgs(convergence_data):
    # graph 1: convergence ratio
    convergence_stats = [[], []]  # [avgs, ratios]
    for data_set in convergence_data:
        avg_convergence = np.sum(data_set) / float(len(data_set))
        number_converged_arr = np.array(data_set) < 50
        convergence_ratio = np.sum(number_converged_arr) / float(len(data_set)) * 100

        convergence_stats[0].append(avg_convergence)
        convergence_stats[1].append(convergence_ratio)

    return convergence_stats


def get_average_recall_ratios(distinct_patterns_data):
    # graph 2: avg. distinct patterns recalled, and relative to set size, i.e. recall ratio
    avg_recall_ratios = []
    stds = []
    for data_set in distinct_patterns_data:
        avg_recalled_patterns = np.sum(data_set).astype(np.float32) / float(len(data_set))
        avg_recall_ratios.append(avg_recalled_patterns)

        stds.append(get_standard_deviation(avg_recalled_patterns, data_set))
    return [avg_recall_ratios, stds]


def get_standard_deviation(avg_value, values):
    # print "avg:", avg_value
    # print "values:", values
    std = 0
    for value in values:
        std += (value-avg_value) ** 2.0
    return np.sqrt(std / float(len(values) - 1))  # -1 degree of freedom


def get_standard_deviation_from_values(values):
    avg_value = np.sum(values) / float(len(values))
    std = 0
    for value in values:
        std += (value-avg_value) ** 2
    return np.sqrt(std / float(len(values) - 1))  # -1 degree of freedom
