
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

filename = 'log.txt'  # log_file_path
print get_data_from_log_file(filename)