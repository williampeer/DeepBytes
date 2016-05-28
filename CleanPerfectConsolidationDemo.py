from DataWrapper import training_patterns_associative, training_patterns_heterogeneous
import NeocorticalMemoryConsolidation
import Tools


def generate_training_set(subset_size, current_training_patterns, patterns_per_output, distortion_P):
    training_patterns = []
    for i in range(5*subset_size):
        current_output = current_training_patterns[i][1]
        for j in range(patterns_per_output):
            training_patterns.append([Tools.flip_bits_f(current_output, distortion_P), current_output])

    return training_patterns


def run_trials_for_patterns_per_output(patterns_per_output):
    # 20 trials per set size, 10 patterns per chaotically recalled output:
    for round_ctr in range(20):
        for set_size_ctr in range(2, 6):
            init_str = 'Performing perfect neocortical memory consolidation according to proposed distortion scheme. ' \
                       +'. Suite round#'+str(round_ctr)+'. Set size ='+str(set_size_ctr)+'.'
            print init_str
            Tools.append_line_to_log(init_str)

            training_set_10 = generate_training_set(set_size_ctr, training_patterns_associative,
                                                    patterns_per_output=patterns_per_output, distortion_P=0.1)
            ann.reset()
            for i in range(100):  # training iterations
                ann.train(training_set_10)
            g_10 = NeocorticalMemoryConsolidation. \
                evaluate_goodness_of_fit(ann, training_patterns_associative[:2*set_size_ctr])
            res_10_str = str(patterns_per_output)+' patterns per output, P=0.1, goodness of fit, g='+str(g_10)
            print res_10_str
            Tools.append_line_to_log(res_10_str)


def run_trials_for_patterns_per_output_on_subsets_sequential(patterns_per_output, distortion_P):
    # 20 trials per set size, 10 patterns per chaotically recalled output:
    for round_ctr in range(20):
        for set_size_ctr in range(2, 6):
            init_str = 'Performing perfect neocortical memory consolidation according to proposed distortion scheme' \
                       'for SUBSETS, i.e. with catastrophic interference. '+ \
                       '. Suite round#'+str(round_ctr)+'. Set size ='+str(set_size_ctr)+'.'
            print init_str
            Tools.append_line_to_log(init_str)

            ann.reset()
            training_set_10 = generate_training_set(
                set_size_ctr, training_patterns_associative, patterns_per_output=patterns_per_output,
                distortion_P=distortion_P)
            for subset_ctr in range(5):
                training_subset_10 = training_set_10[subset_ctr * set_size_ctr * patterns_per_output:
                    (subset_ctr + 1) * set_size_ctr * patterns_per_output]
                for i in range(15):  # training iterations
                    ann.train(training_subset_10)
            g_10 = NeocorticalMemoryConsolidation. \
                evaluate_goodness_of_fit(ann, training_patterns_associative[:2 * set_size_ctr])
            res_10_str = str(i+1) + ' training iterations, ' + str(patterns_per_output) + \
                         ' patterns per output, P='+str(distortion_P)+', goodness of fit, g=' + str(g_10)
            print res_10_str
            Tools.append_line_to_log(res_10_str)


ann = NeocorticalMemoryConsolidation.NeocorticalNetwork(49, 30, 49, 0.01, 0.9)
# run_trials_for_patterns_per_output(10)
# run_trials_for_patterns_per_output(20)

run_trials_for_patterns_per_output_on_subsets_sequential(patterns_per_output=7, distortion_P=0.05)
run_trials_for_patterns_per_output_on_subsets_sequential(patterns_per_output=7, distortion_P=0.1)
run_trials_for_patterns_per_output_on_subsets_sequential(patterns_per_output=7, distortion_P=0.15)
# run_trials_for_patterns_per_output_on_subsets_sequential(patterns_per_output=10, distortion_P=0.05)
