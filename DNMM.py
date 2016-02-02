import time
import numpy as np
from Tools import binomial_f, uniform_f, show_image_from


def hpc_learn_patterns_wrapper(hpc, patterns, max_training_iterations):
    print "Commencing learning of", len(patterns), "I/O patterns."
    time_start_overall = time.time()
    iter_ctr = 0
    learned_all = False
    while not learned_all and iter_ctr < max_training_iterations:
        p_ctr = 0
        for [input_pattern, output_pattern] in patterns:
            # Neuronal turnover, setting input and output in the hpc network.
            setup_start = time.time()
            hpc.setup_pattern(input_pattern, output_pattern)
            setup_end = time.time()
            print "Setup took:", "{:6.3f}".format(setup_end-setup_start), "seconds."
            # if setup_end-setup_start < 0.120:
                # hpc.print_activation_values()
                # hpc.print_ca3_info()
                # hpc.print_activation_values_and_weights()
            hpc.print_last_halves_of_activation_values_sums()
            hpc.print_activation_values_sum()

            # one iteration of learning using Hebbian learning
            time_before = time.time()
            hpc.learn()
            time_after = time.time()
            print "Iterated over pattern", p_ctr, "in", \
                "{:7.3f}".format(time_after - time_before), "seconds."
            # hpc.print_activation_values_sum()
            p_ctr += 1

        learned_all = True
        print "Attempting to recall patterns..."
        for pattern_index in range(len(patterns)):
            print "Recalling pattern #", pattern_index
            hpc.setup_input(patterns[pattern_index][0])
            hpc.recall()
            hpc.recall()
            hpc.recall()
            out_values_row = hpc.output_values.get_value()[0]
            cur_p_row = patterns[pattern_index][1][0]
            # print "outvals:", out_values
            # print "curp", cur_p
            for el_index in range(len(cur_p_row)):
                if out_values_row[el_index] != cur_p_row[el_index]:
                    learned_all = False
                    print "Patterns are not yet successfully learned. Learning more..."
                    # print "Displaying intermediary results... (output, target)"
                    # show_image_from(np.asarray([out_values_row], dtype=np.float32))
                    # show_image_from(np.asarray([cur_p_row], dtype=np.float32))
                    print "iter:", iter_ctr
                    break
            if not learned_all:
                break

        iter_ctr += 1
    time_stop_overall = time.time()

    print "Learned", len(patterns), "pattern-associations in ", iter_ctr, "iterations, which took" "{:7.3f}". \
        format(time_stop_overall-time_start_overall), "seconds."


def hpc_chaotic_recall_wrapper(hpc, display_images_of_stable_output, recall_iterations):
    time_the_beginning_of_time = time.time()
    time_before = time.time()
    cur_iters = 0
    random_input = uniform_f(1, hpc.dims[0]) * 2 - np.ones_like(hpc.input_values, dtype=np.float32)
    hpc.setup_input(random_input)
    hpc_extracted_pseudopatterns = []
    while cur_iters < recall_iterations:
        [cur_iters_term, found_stable_output, output] = hpc.recall_until_stability_criteria(
                should_display_image=display_images_of_stable_output, max_iterations=recall_iterations-cur_iters)
        cur_iters += cur_iters_term

        if found_stable_output:
            hpc_extracted_pseudopatterns.append(output)

        time_after = time.time()
        prop_time_until_stable = time_after - time_before

        print "Propagation time until stability:", "{:6.3f}".format(prop_time_until_stable), "seconds."
        print "t =", cur_iters
        time_before = time.time()
    print "Total chaotic recall time:", "{:6.3f}".format(time.time()-time_the_beginning_of_time), "seconds."
    return hpc_extracted_pseudopatterns


def generate_pseodupatterns_II(dim, hpc_extracted_pseudopatterns, reverse_P, set_size):
    extracted_set_size = len(hpc_extracted_pseudopatterns)
    pseudopatterns_II = []
    pseudopattern_ctr = 0
    while pseudopattern_ctr < set_size:
        pattern = hpc_extracted_pseudopatterns[pseudopattern_ctr % extracted_set_size]
        # q=1-p because we're flipping the sign of the ones that are not flipped.
        reverse_vector = binomial_f(1, dim, (1-reverse_P))
        reverse_vector = reverse_vector * 2 - np.ones_like(reverse_vector)
        pseudopatterns_II.append(pattern * reverse_vector)
    return pseudopatterns_II