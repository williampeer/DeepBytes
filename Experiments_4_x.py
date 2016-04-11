import time
from HPCWrappers import hpc_learn_patterns_wrapper, hpc_chaotic_recall_wrapper, generate_pseudopattern_II_hpc_outputs
from Tools import set_contains_pattern, get_pattern_correlation, save_experiment_4_1_results, save_images_from
import Tools
import numpy as np

# next experiment output image:
next_experiment_im = [[-1, 1] * 24]
next_experiment_im[0].append(-1)


def experiment_4_x_1(hpc, training_set_size, original_training_patterns):
    hippocampal_chaotic_recall_patterns = []
    random_ins = []

    for train_set_num in range(5):  # always five training sets
        current_set_hipp_chaotic_recall, current_set_random_ins = \
            training_and_recall_hpc_helper(hpc, training_set_size, train_set_num, original_training_patterns)
        hippocampal_chaotic_recall_patterns += current_set_hipp_chaotic_recall
        random_ins.append(current_set_random_ins)

    # show_image_from(np.asarray(next_experiment_im).astype(np.float32))
    return [hippocampal_chaotic_recall_patterns, random_ins]


def training_and_recall_hpc_helper(hpc, training_set_size, train_set_num, original_training_patterns):
    hippocampal_chaotic_recall_patterns = []
    random_ins = []
    # Setup current training patterns:
    training_set = original_training_patterns[training_set_size * train_set_num : training_set_size +
                                                                           train_set_num*training_set_size]

    # print "Performing neuronal turnover in DG for", hpc._turnover_rate * 100, "% of the neurons.."
    # t0 = time.time()
    # hpc.neuronal_turnover_dg()
    # t1 = time.time()
    # print "Neuronal turnover completed in", "{:7.3f}".format(t1-t0), "seconds."
    # hpc.re_wire_fixed_input_to_ec_weights()
    print "Learning patterns in training set..."
    hpc_learn_patterns_wrapper(hpc, patterns=training_set, max_training_iterations=50)  # when training is fixed,
    # convergence should occur after one or two iterations?

    # extract by chaotic recall:
    chaotic_recall_iters = 300
    print "Recalling patterns for", chaotic_recall_iters, "time-steps by chaotic recall..."
    t2 = time.time()
    # hpc.reset_zeta_and_nu_values()
    [patterns_extracted_for_current_set, random_in] = \
        hpc_chaotic_recall_wrapper(hpc, display_images_of_stable_output=False, recall_iterations=chaotic_recall_iters)
    for pat in patterns_extracted_for_current_set:
        if not set_contains_pattern(hippocampal_chaotic_recall_patterns, pat):
            hippocampal_chaotic_recall_patterns.append(pat)  # append unique pattern
            random_ins.append(random_in)
    t3 = time.time()
    print "Set size for hippocampal_chaotic_recall_patterns:", len(hippocampal_chaotic_recall_patterns)
    print "Chaotic recall completed in", "{:8.3f}".format(t3-t2), "seconds, for t=300."
    Tools.append_line_to_log("Recalled " + str(len(hippocampal_chaotic_recall_patterns)) +
                             " distinct patterns by chaotic recall.")

    # Use this to debug the current model:
    # learned_ctr = 0
    # for pat in training_set:
    #     hpc.setup_input(pat[0])
    #     print "Recalling pattern #", learned_ctr
    #     # show_image_from(np.asarray(next_experiment_im).astype(np.float32))
    #     hpc.recall()
    #     show_image_from(hpc.output_values.get_value())
    #     learned_ctr += 1

    return [hippocampal_chaotic_recall_patterns, random_ins]


def experiment_4_x_2(hpc, ann, training_set_size, original_training_patterns):
    pseudopattern_set_size = 20  # this should be set to 20. debugging mode: small value.

    # Generate pseudopatterns:
    chaotically_recalled_patterns = []
    pseudopatterns_I = []
    pseudopatterns_II = []

    for train_set_num in range(5):  # always five training sets
        current_set_hipp_chaotic_recall, current_set_random_ins = \
            training_and_recall_hpc_helper(hpc, training_set_size, train_set_num, original_training_patterns)

        for curr_chaotic_p in current_set_hipp_chaotic_recall:
            chaotically_recalled_patterns.append(curr_chaotic_p)

        current_pseudopatterns_I = []
        current_pseudopatterns_II = []

        # using the same temporarily constant random input and output as in the chaotic recall session:
        if len(current_set_random_ins) > 0:
            # for pattern_ctr in range(pseudopattern_set_size):
            #     current_random_in = current_set_random_ins[pattern_ctr % len(current_set_random_ins)]
            #     current_hipp_recall = current_set_hipp_chaotic_recall[pattern_ctr % len(current_set_hipp_chaotic_recall)]
            #     current_pseudopatterns_I.append([current_random_in, current_hipp_recall])

            # Reflecting HPC config.:
            for pattern_ctr in range(pseudopattern_set_size):
                # generate random input:
                random_hpc_input = Tools.binomial_f(1, hpc.dims[0], 0.5) * 2 - np.ones_like(hpc.input_values,
                                                                                            dtype=np.float32)
                hpc.setup_input(random_hpc_input)
                hpc.recall()
                # hpc_output = hpc.output_values.get_value()  # first output from random input.
                # gets the first stable or timed out output:
                _, _, hpc_output = hpc.recall_until_stability_criteria(False, 300)
                current_pseudopatterns_I.append([random_hpc_input, hpc_output])

            pseudopatterns_I += current_pseudopatterns_I
            # ann.train(current_pseudopatterns_I)

            # current_pseudopatterns_II_inputs = \
            #     generate_pseudopattern_II_hpc_outputs(hpc.dims[0], current_set_hipp_chaotic_recall, 0.5,
            #                                           pseudopattern_set_size)
            # for current_pseudo_input in current_pseudopatterns_II_inputs:
            #     current_pseudopatterns_II += [ann.get_IO(current_pseudo_input)]
            #
            # pseudopatterns_II += current_pseudopatterns_II
            ann.train(current_pseudopatterns_I)
            # ann.train(current_pseudopatterns_II)

    # ann.train(pseudopatterns_I)
    # ann.train(pseudopatterns_II)

    save_experiment_4_1_results(hpc, chaotically_recalled_patterns, "exp_1_before2")

    # Attempt to recall using the entire DNMM:
    sum_corr = 0.
    corr_ctr = 0.
    neocortically_recalled_pairs = []
    for [target_in, target_out] in original_training_patterns:
        obtained_in, obtained_out = ann.get_IO(target_in)
        sum_corr += get_pattern_correlation(target_out, obtained_out)
        corr_ctr += 1
        neocortically_recalled_pairs.append([obtained_in, obtained_out])
    g = sum_corr / corr_ctr
    print "goodness of fit, g=", g

    return [pseudopatterns_I, pseudopatterns_II, neocortically_recalled_pairs, ann, g]
