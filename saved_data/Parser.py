

# returns: [ experiment: [set_size, #sets, DGW, 5x tuples: [iters_to_convergence, distinct_patterns_recalled]]]
def get_data_from_log_file(filename):
    log_file = file(filename, 'r')

    contents = log_file.read()
    log_file.close()

    lines = contents.split('\n')
    all_data = []
    for experiment_ctr in range(len(lines)/11):
        experiment_data = []

        sep = lines[11*experiment_ctr].split('x')
        set_size = int(sep[0][len(sep[0])-1])
        training_sets = int(sep[1][0])

        words = lines[experiment_ctr * 11].split()
        dg_weighting = words[-1][:len(words[-1])-1]

        experiment_data.append(set_size)
        experiment_data.append(training_sets)
        experiment_data.append(float(dg_weighting))
        for i in range(5):
            convergence_line_words = lines[11 * experiment_ctr + 1 + 2 * i].split()
            patterns_recalled_line_words = lines[11 * experiment_ctr + 2 + 2 * i].split()

            iterations_before_convergence = int(convergence_line_words[2])
            distinct_patterns_recalled = int(patterns_recalled_line_words[1])
            experiment_data.append([iterations_before_convergence, distinct_patterns_recalled])

        all_data.append(experiment_data)

    return all_data


def get_convergence_and_distinct_patterns_from_log_v1(parsed_data):
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

    return [experiment_data_convergence, experiment_data_distinct_patterns]


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
