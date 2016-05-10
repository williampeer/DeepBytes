

# returns: [ experiment: [set_size, #sets, DGW, 5x tuples: [iters_to_convergence, distinct_patterns_recalled]]]
def get_data_from_log_file_hpc_chaotic_recall(filename, lines_of_learning_spam):
    log_file = file(filename, 'r')
    lines_per_exp = lines_of_learning_spam + 3

    contents = log_file.read()
    log_file.close()

    lines = contents.split('\n')
    all_data = []
    for experiment_ctr in range(len(lines)/lines_per_exp):
        experiment_data = []

        first_line_in_experiment_split_on_x = lines[lines_per_exp*experiment_ctr].split('x')
        # get last symbol of first sub-string
        set_size = int(first_line_in_experiment_split_on_x[0][len(first_line_in_experiment_split_on_x[0])-1])
        training_sets = int(first_line_in_experiment_split_on_x[1][0])

        words = lines[experiment_ctr * lines_per_exp].split()
        dg_weighting = words[-1][:len(words[-1])-1]

        perfect_recall_rate = float(lines[lines_per_exp * experiment_ctr + (lines_per_exp-2)].split()[-1])
        num_of_spurious_patterns = int(lines[lines_per_exp * experiment_ctr + (lines_per_exp-1)].split()[-1])

        experiment_data.append(set_size)
        experiment_data.append(training_sets)
        experiment_data.append(int(dg_weighting))
        experiment_data.append(perfect_recall_rate)
        experiment_data.append(num_of_spurious_patterns)

        all_data.append(experiment_data)

    return all_data


log_filename_hpc_chaotic_local = 'log-local-hpc-chaotic.txt'
hpc_chaotic_parsed_data = get_data_from_log_file_hpc_chaotic_recall(log_filename_hpc_chaotic_local, 5)
print hpc_chaotic_parsed_data

log_filename_hpc_global = 'log-hpc-recall-schemes.txt'
hpc_chaotic_global_parsed_data = get_data_from_log_file_hpc_chaotic_recall(log_filename_hpc_global, 15)
print hpc_chaotic_global_parsed_data