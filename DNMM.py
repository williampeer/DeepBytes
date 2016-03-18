import time
import numpy as np
from Tools import binomial_f, uniform_f, show_image_from
from HPC import HPC


def hpc_learn_patterns_wrapper(hpc, training_patterns, max_training_iterations):
    test_hpc = HPC([49, 240, 1600, 480, 49],
          0.67, 0.25, 0.04,  # connection rates: (in_ec, ec_dg, dg_ca3)
          0.10, 0.01, 0.04,  # firing rates: (ec, dg, ca3)
          0.7, 1.0, 0.1, 0.5,  # gamma, epsilon, nu, turnover rate
          0.10, 0.95, 0.8, 2.0)  # k_m, k_r, a_i, alpha. alpha is 2 in 4.1

    print "Commencing learning of", len(training_patterns), "I/O patterns."
    time_start_overall = time.time()
    iter_ctr = 0
    learned_all = False
    # while iter_ctr < 3:
    while not learned_all and iter_ctr < max_training_iterations:
        p_ctr = 0
        for [input_pattern, output_pattern] in training_patterns:
            # one iteration of learning using Hebbian learning
            time_before = time.time()
            hpc.neuronal_turnover_dg()
            hpc.learn(input_pattern, output_pattern)
            hpc.learn(input_pattern, output_pattern)
            time_after = time.time()
            print "Iterated over pattern", p_ctr, "in", \
                "{:7.3f}".format(time_after - time_before), "seconds."
            # hpc.print_activation_values_sum()
            p_ctr += 1

        learned_all = True
        print "Attempting to recall patterns..."
        for pattern_index in range(len(training_patterns)):
            print "Recalling pattern #", pattern_index
            test_hpc.in_ec_weights = hpc.in_ec_weights
            test_hpc.ec_dg_weights = hpc.ec_dg_weights
            test_hpc.ec_ca3_weights = hpc.ec_ca3_weights
            test_hpc.dg_ca3_weights = hpc.dg_ca3_weights
            test_hpc.ca3_ca3_weights = hpc.ca3_ca3_weights
            test_hpc.ca3_out_weights = hpc.ca3_out_weights

            test_hpc.recall(training_patterns[pattern_index][0])
            test_hpc.recall(training_patterns[pattern_index][0])
            test_hpc.recall(training_patterns[pattern_index][0])

            out_values_row = test_hpc.output_values.get_value()[0]
            cur_p_row = training_patterns[pattern_index][1][0]
            # print "outvals:", out_values
            # print "curp", cur_p
            for el_index in range(len(cur_p_row)):
                if out_values_row[el_index] != cur_p_row[el_index]:
                    learned_all = False
                    print "Patterns are not yet successfully learned. Learning more..."
                    print "Displaying intermediary result(s)... (out, target)"
                    show_image_from(np.asarray([out_values_row], dtype=np.float32))
                    show_image_from(np.asarray([cur_p_row], dtype=np.float32))
                    print "iter:", iter_ctr
                    break
            if not learned_all:
                break

        iter_ctr += 1
    time_stop_overall = time.time()

    print "Learned", len(training_patterns), "pattern-associations in ", iter_ctr, "iterations, which took" "{:8.3f}". \
        format(time_stop_overall-time_start_overall), "seconds."


def hpc_chaotic_recall_wrapper(hpc, display_images_of_stable_output, recall_iterations):
    time_start = time.time()

    random_input = binomial_f(1, hpc.dims[0], 0.5) * 2 - np.ones_like(hpc.input_values, dtype=np.float32)
    hpc.set_input(random_input)
    hpc.fire_in_ec_wrapper()  # in order to omit this in the chaotic recall-loop

    hpc_extracted_pseudopatterns = []
    cur_iters = 0
    time_current_loop = time.time()
    while cur_iters < recall_iterations:
        [cur_iters_term, found_stable_output, output] = hpc.recall_until_stability_criteria(
                should_display_image=display_images_of_stable_output, max_iterations=recall_iterations-cur_iters)
        cur_iters += cur_iters_term

        if found_stable_output:
            hpc_extracted_pseudopatterns.append(output)

        time_after = time.time()
        prop_time_until_stable = time_after - time_current_loop

        print "Propagation time until stability:", "{:6.3f}".format(prop_time_until_stable), "seconds."
        print "t =", cur_iters
        time_current_loop = time.time()
    print "Total chaotic recall time:", "{:6.3f}".format(time.time()-time_start), "seconds."
    return [hpc_extracted_pseudopatterns, random_input]


def generate_pseudopattern_II_hpc_outputs(dim, hpc_extracted_pseudopatterns, reverse_P, set_size):
    extracted_set_size = len(hpc_extracted_pseudopatterns)
    pseudopatterns_II = []
    pseudopattern_ctr = 0
    while pseudopattern_ctr < set_size:
        pattern = hpc_extracted_pseudopatterns[pseudopattern_ctr % extracted_set_size]
        # q=1-p because we're flipping the sign of the ones that are not flipped.
        reverse_vector = binomial_f(1, dim, (1-reverse_P))
        reverse_vector = reverse_vector * 2 - np.ones_like(reverse_vector)
        pseudopatterns_II.append(pattern * reverse_vector)
        pseudopattern_ctr += 1
    return pseudopatterns_II