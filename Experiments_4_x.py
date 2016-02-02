import time
from DNMM import hpc_learn_patterns_wrapper, hpc_chaotic_recall_wrapper, generate_pseodupatterns_II
from Tools import show_image_from, set_contains_pattern, get_pattern_correlation

# next experiment output image:
next_experiment_im = [[-1, 1] * 24]
next_experiment_im[0].append(-1)


def experiment_4_x_1(hpc, training_set_size, original_training_patterns):
    hippocampal_chaotic_recall_patterns = []

    for train_set_num in range(5):  # always five training sets
        # Setup current training patterns:
        training_set = original_training_patterns[training_set_size * train_set_num : training_set_size +
                                                                               train_set_num*training_set_size]

        print "Performing neuronal turnover in DG for", hpc._turnover_rate * 100, "% of the neurons.."
        t0 = time.time()
        hpc.neuronal_turnover_dg()
        t1 = time.time()
        print "Neuronal turnover completed in", "{:7.3f}".format(t1-t0), "seconds."
        print "Learning patterns in training set..."
        hpc_learn_patterns_wrapper(hpc, patterns=training_set, max_training_iterations=15)

        # extract by chaotic recall:
        # print "Recalling patterns for 300 time-steps by chaotic recall..."
        # t2 = time.time()
        # patterns_extracted_for_current_set = \
        #     hpc_chaotic_recall_wrapper(hpc, display_images_of_stable_output=False, recall_iterations=300)
        # for pat in patterns_extracted_for_current_set:
        #     if not set_contains_pattern(hippocampal_chaotic_recall_patterns, pat):
        #         hippocampal_chaotic_recall_patterns.append(pat)  # append unique pattern
        # t3 = time.time()
        # print "Chaotic recall completed in", "{:8.3f}".format(t3-t2), "seconds, for t=300."

        # Use this to debug the current model:
        learned_ctr = 0
        for pat in training_set:
            hpc.setup_input(pat[0])
            print "Recalling pattern #", learned_ctr
            # show_image_from(np.asarray(next_experiment_im).astype(np.float32))
            hpc.recall()
            show_image_from(hpc.output_values.get_value())
            learned_ctr += 1

    # show_image_from(np.asarray(next_experiment_im).astype(np.float32))
    return hippocampal_chaotic_recall_patterns


def experiment_4_x_2(hpc, ann, training_set_size, original_training_patterns):
    pseudopattern_set_size = 20

    # Generate pseudopatterns:
    pseudopatterns_I = []
    for i in range(pseudopattern_set_size):
        pseudopatterns_I.append(ann.get_pseudopattern_I())

    chaotically_recalled_patterns = experiment_4_x_1(training_set_size, original_training_patterns)
    pseudopatterns_II = generate_pseodupatterns_II(hpc.dims[0], chaotically_recalled_patterns, 0.5,
                                                   pseudopattern_set_size)

    # Train Neocortical network on them:
    training_set = pseudopatterns_I + pseudopatterns_II
    for training_pattern in training_set:
        ann.train([training_pattern, training_pattern])  # learns the identity map.

    # Attempt to recall using the entire DNMM:
    sum_corr = 0.
    for original_pattern in original_training_patterns:
        ann.feed_forward(original_pattern)
        sum_corr += get_pattern_correlation(original_pattern, ann._out.get_value())
    g = sum_corr / len(sum_corr)
    print "goodness of fit, g=", g

    return [pseudopatterns_I, pseudopatterns_II, ann, hpc, g]