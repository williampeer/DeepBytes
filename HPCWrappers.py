import time
import numpy as np
import Tools
from HPC import HPC


def hpc_learn_patterns_wrapper(hpc, patterns, max_training_iterations):

    test_hpc = HPC([hpc.dims[0], hpc.dims[1], hpc.dims[2], hpc.dims[3], hpc.dims[4]],
                   hpc.connection_rate_input_ec, hpc.PP, hpc.MF,  # connection rates: (in_ec, ec_dg, dg_ca3)
                   hpc.firing_rate_ec, hpc.firing_rate_dg, hpc.firing_rate_ca3,  # firing rates: (ec, dg, ca3)
                   hpc._gamma, hpc._epsilon, hpc._nu, hpc._turnover_rate,  # gamma, epsilon, nu, turnover rate
                   hpc._k_m, hpc._k_r, hpc._a_i.get_value()[0][0], hpc._alpha, hpc._weighting_dg,  # k_m, k_r, a_i, alpha. alpha is 2 in 4.1
                   _ASYNC_FLAG=hpc._ASYNC_FLAG, _TURNOVER_MODE=hpc._TURNOVER_MODE)

    print "Commencing learning of", len(patterns), "I/O patterns."
    time_start_overall = time.time()
    iter_ctr = 0
    learned_all = False

    # scope: for every new set
    if hpc._TURNOVER_MODE == 0:
        neuronal_turnover_helper(hpc)

    while not learned_all and iter_ctr < max_training_iterations:
        p_ctr = 0

        # scope: for every training set iteration
        if hpc._TURNOVER_MODE == 1:
            neuronal_turnover_helper(hpc)

        for [input_pattern, output_pattern] in patterns:
            setup_start = time.time()
            hpc.setup_pattern(input_pattern, output_pattern)
            setup_end = time.time()
            print "Setup took:", "{:6.3f}".format(setup_end-setup_start), "seconds."

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
        # test_hpc.reset_zeta_and_nu_values()
        for pattern_index in range(len(patterns)):
            print "Recalling pattern #", pattern_index
            test_hpc = Tools.set_to_equal_parameters(hpc, test_hpc)
            test_hpc.setup_input(patterns[pattern_index][0])

            test_hpc.recall()
            test_hpc.recall()
            test_hpc.recall()

            out_values_row = test_hpc.output_values.get_value()[0]
            cur_p_row = patterns[pattern_index][1][0]
            for el_index in range(len(cur_p_row)):
                if out_values_row[el_index] != cur_p_row[el_index]:
                    learned_all = False
                    print "Patterns are not yet successfully learned. Learning more..."
                    # print "Displaying intermediary result(s)..."
                    # show_image_from(np.asarray([out_values_row], dtype=np.float32))
                    # show_image_from(np.asarray([cur_p_row], dtype=np.float32))
                    print "iter:", iter_ctr
                    break
            if not learned_all:
                break

        iter_ctr += 1
    time_stop_overall = time.time()

    print "Terminated learning", len(patterns), "pattern-associations in ", iter_ctr, "iterations, which took" "{:8.3f}". \
        format(time_stop_overall-time_start_overall), "seconds."
    Tools.append_line_to_log("Convergence after " + str(iter_ctr) + " iterations. Turnover: " +
                             str(hpc._turnover_rate) + ". DG-weighting: " + str(hpc._weighting_dg))


def hpc_chaotic_recall_wrapper(hpc, display_images_of_stable_output, recall_iterations):
    time_the_beginning_of_time = time.time()
    time_before = time.time()
    cur_iters = 0
    random_input = 2 * Tools.binomial_f(1, hpc.dims[0], 0.5) - np.ones_like(hpc.input_values, dtype=np.float32)
    hpc.setup_input(random_input)
    hpc_extracted_pseudopatterns = []
    while cur_iters < recall_iterations:
        [cur_iters_term, found_stable_output, output] = hpc.recall_until_stability_criteria(
                should_display_image=display_images_of_stable_output, max_iterations=recall_iterations-cur_iters)
        cur_iters += cur_iters_term

        if found_stable_output:
            hpc_extracted_pseudopatterns.append(output)
            random_input = 2 * Tools.binomial_f(1, hpc.dims[0], 0.5) - np.ones_like(hpc.input_values, dtype=np.float32)
            hpc.setup_input(random_input)

        time_after = time.time()
        prop_time_until_stable = time_after - time_before

        print "Propagation time until stability:", "{:6.3f}".format(prop_time_until_stable), "seconds."
        print "t =", cur_iters
        time_before = time.time()
    print "Total chaotic recall time:", "{:6.3f}".format(time.time()-time_the_beginning_of_time), "seconds."
    return [hpc_extracted_pseudopatterns, random_input]


def neuronal_turnover_helper(hpc):
    print "Performing neuronal turnover..."
    t0 = time.time()
    hpc.neuronal_turnover_dg()
    t1 = time.time()
    print "Completed neuronal turnover for " + str(hpc._turnover_rate * 100) + " % of the neurons in " + \
          "{:6.3f}".format(t1-t0), "seconds."


def learn_patterns_for_i_iters_hpc_wrapper(hpc, patterns, num_of_iters):
    print "Commencing learning of", len(patterns), "I/O patterns."
    time_start_overall = time.time()

    for i in range(num_of_iters):
        # scope: for every training set iteration
        if hpc._TURNOVER_MODE == 1:
            neuronal_turnover_helper(hpc)

        for [input_pattern, output_pattern] in patterns:
            # setup_start = time.time()
            hpc.setup_pattern(input_pattern, output_pattern)
            # setup_end = time.time()
            # print "Setup took:", "{:6.3f}".format(setup_end-setup_start), "seconds."

            # one iteration of learning using Hebbian learning
            hpc.learn()

    time_stop_overall = time.time()
    print "Terminated learning", len(patterns), "pattern-associations in ", num_of_iters, \
        "iterations, which took" "{:8.3f}".format(time_stop_overall-time_start_overall), "seconds."
    Tools.append_line_to_log("Learned for " + str(num_of_iters) + " iterations. Turnover: " + str(hpc._turnover_rate) +
                             ". DG-weighting: " + str(hpc._weighting_dg))


def hpc_generate_pseudopatterns_I_wrapper(hpc, num_of_pseudopatterns):
    pattern_length = hpc.input_values.get_value().shape[1]
    pseudopatterns = []
    for i in range(num_of_pseudopatterns):
        random_input = 2 * Tools.binomial_f(1, pattern_length, 0.5) - np.ones((1, pattern_length), dtype=np.float32)
        corresponding_output = hpc.propagate_until_stable(random_input)
        pseudopatterns.append([random_input, corresponding_output])
    return pseudopatterns


def hpc_generate_pseudopatterns_II_wrapper(hpc, num_of_pseudopatterns, chaotically_recalled_patterns, flip_P):
    pseudopatterns = []
    distinct_chaotically_recalled_patts = []
    for p in chaotically_recalled_patterns:
        if not Tools.set_contains_pattern(distinct_chaotically_recalled_patts, p):
            distinct_chaotically_recalled_patts.append(p)
    if len(chaotically_recalled_patterns) == 0:
        print "ERROR: len(chaotically_recalled_patterns) is 0"
        return []

    chaotic_set_size = len(distinct_chaotically_recalled_patts)
    for i in range(num_of_pseudopatterns):
        current_chaotically_recalled_pattern = distinct_chaotically_recalled_patts[i % chaotic_set_size]
        current_input = Tools.flip_bits_f(current_chaotically_recalled_pattern, flip_P)
        pseudopatterns.append([current_input, hpc.propagate_until_stable(current_input)])
    return pseudopatterns


def hpc_generate_pseudopatterns_II_recall_i_iters_wrapper(hpc, num_of_pseudopatterns, chaotically_recalled_patterns, flip_P):
    pseudopatterns = []
    if len(chaotically_recalled_patterns) == 0:
        print "ERROR: len(chaotically_recalled_patterns) is 0"
        return []

    chaotic_set_size = len(chaotically_recalled_patterns)
    for i in range(num_of_pseudopatterns):
        current_chaotically_recalled_pattern = chaotically_recalled_patterns[i % chaotic_set_size]
        current_input = Tools.flip_bits_f(current_chaotically_recalled_pattern, flip_P)[0]
        pseudopatterns.append([current_input, hpc.recall_for_i_iters_with_input(current_input, 15)])
    return pseudopatterns


def hpc_generate_pseudopatterns_I_recall_i_iters_wrapper(hpc, num_of_pseudopatterns, num_of_iters):
    pseudopatterns = []
    for i in range(num_of_pseudopatterns):
        rand_in, output = hpc.recall_for_i_iters(should_display_image=False, num_of_iterations=num_of_iters)
        pseudopatterns.append([rand_in, output])
    return pseudopatterns


def hpc_generate_pseudopatterns_I_recall_i_iters_wrapper_random_stream(hpc, num_of_pseudopatterns, num_of_iters):
    pseudopatterns = []
    for i in range(num_of_pseudopatterns):
        rand_in, output = hpc.recall_for_i_iters_random_stream_in(
            should_display_image=False, num_of_iterations=num_of_iters)
        pseudopatterns.append([rand_in, output])
    return pseudopatterns