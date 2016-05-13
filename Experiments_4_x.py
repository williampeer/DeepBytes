import time, HPCWrappers
from HPCWrappers import hpc_learn_patterns_wrapper, hpc_chaotic_recall_wrapper
import Tools
from HPC import HPC

# next experiment output image:
next_experiment_im = [[-1, 1] * 24]
next_experiment_im[0].append(-1)


def experiment_4_x_1(hpc, training_set_size, original_training_patterns):
    Tools.append_line_to_log("INIT. EXPERIMENT #" + str(Tools.get_experiment_counter()) + ". Type: 4_x_1" +
                             ": ASYNC-flag:" + str(hpc._ASYNC_FLAG) + ". " + str(training_set_size) + "x5. " +
                             "Turnover mode: " + str(hpc._TURNOVER_MODE) + ". Turnover rate:" +
                             str(hpc._turnover_rate) + ", DG-weighting: " + str(hpc._weighting_dg) + ".")

    hippocampal_chaotic_recall_patterns = []
    random_ins = []

    for train_set_num in range(5):  # always five training sets
        current_set_hipp_chaotic_recall, current_set_random_ins = \
            training_and_recall_hpc_helper(hpc, training_set_size, train_set_num, original_training_patterns)
        hippocampal_chaotic_recall_patterns.append(current_set_hipp_chaotic_recall)
        random_ins.append(current_set_random_ins)

    # show_image_from(np.asarray(next_experiment_im).astype(np.float32))
    return [hippocampal_chaotic_recall_patterns, random_ins]


def training_and_recall_hpc_helper(hpc, training_set_size, train_set_num, original_training_patterns):
    hippocampal_chaotic_recall_patterns = []
    random_ins = []
    # Setup current training patterns:
    training_set = original_training_patterns[training_set_size * train_set_num : training_set_size +
                                                                           train_set_num*training_set_size]

    print "Learning patterns in training set..."
    hpc_learn_patterns_wrapper(hpc, patterns=training_set, max_training_iterations=50)  # when training is fixed,
    # convergence should occur after one or two iterations?

    # extract by chaotic recall:
    chaotic_recall_iters = 300
    print "Recalling patterns for", chaotic_recall_iters, "time-steps by chaotic recall..."
    t2 = time.time()
    [patterns_extracted_for_current_set, random_in] = \
        hpc_chaotic_recall_wrapper(hpc, display_images_of_stable_output=False, recall_iterations=chaotic_recall_iters)
    for pat in patterns_extracted_for_current_set:
        # if not set_contains_pattern(hippocampal_chaotic_recall_patterns, pat):
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
    Tools.append_line_to_log("INIT. EXPERIMENT #" + str(Tools.get_experiment_counter()) + ". Type: 4_x_2" +
                             ": ASYNC-flag:" + str(hpc._ASYNC_FLAG) + ". " + str(training_set_size) + "x5. " +
                             "Turnover mode: " + str(hpc._TURNOVER_MODE) + ". Turnover rate:" +
                             str(hpc._turnover_rate) + ", DG-weighting: " + str(hpc._weighting_dg) + ".")


    pseudopattern_set_size = 20  # this should be set to 20. debugging mode: small value.
    pseudopattern_I_set_size = pseudopattern_set_size/2
    pseudopattern_II_set_size = pseudopattern_set_size - pseudopattern_I_set_size

    chaotically_recalled_patterns = []
    all_rand_ins = []

    pseudopatterns_I = []
    pseudopatterns_II = []

    for train_set_num in range(5):  # always five training sets
        current_set_hipp_chaotic_recall, current_set_random_ins = \
            training_and_recall_hpc_helper(hpc, training_set_size, train_set_num, original_training_patterns)

        for p_ctr in range(len(current_set_hipp_chaotic_recall)):
            # if not Tools.set_contains_pattern(chaotically_recalled_patterns, current_set_hipp_chaotic_recall[p_ctr]):
            chaotically_recalled_patterns.append([current_set_hipp_chaotic_recall[p_ctr]])
            all_rand_ins.append([current_set_random_ins[p_ctr]])

        current_pseudopatterns_I = []
        current_pseudopatterns_II = []

        tmp_p_I_set = []
        if len(current_set_hipp_chaotic_recall) >= pseudopattern_I_set_size:
            tmp_p_I_set += current_set_hipp_chaotic_recall[:pseudopattern_I_set_size]
        else:
            tmp_p_I_set += current_set_hipp_chaotic_recall
            while len(tmp_p_I_set) < pseudopattern_I_set_size:
                print "len(tmp_p_I_set):", len(tmp_p_I_set)
                _, _, cur_p_recallled = hpc.recall_until_stability_criteria(should_display_image=False, max_iterations=300)
                tmp_p_I_set.append(cur_p_recallled)

        # train on currently extracted patterns
        ann_current_training_set = []
        for i in range(len(current_set_hipp_chaotic_recall)):
            ann_current_training_set.append([current_set_random_ins[i], current_set_hipp_chaotic_recall[i]])

        ann.train(ann_current_training_set)

        # generate p_I's; should reverse outputs, and get IO's from ANN as p_I
        pattern_length = len(tmp_p_I_set[0][0])
        for i in range(len(tmp_p_I_set)):
            # flip_bits = np.ones_like(tmp_p_I_set[0], dtype=np.float32) - 2 * Tools.binomial_f(1, pattern_length, 0.5)
            # pattern = tmp_p_I_set[i] * flip_bits  # binomial_f returns a 2-dim. array
            pattern = Tools.flip_bits_f(tmp_p_I_set[i], flip_P=0.5)
            I, O = ann.get_IO(pattern)
            current_pseudopatterns_I.append([I, O])
        # generate p_II's
        while len(current_pseudopatterns_II) < pseudopattern_II_set_size:
            I, O = ann.get_random_IO()
            current_pseudopatterns_II.append([I, O])

        ann.train(current_pseudopatterns_I)
        ann.train(current_pseudopatterns_II)

        pseudopatterns_I += current_pseudopatterns_I
        pseudopatterns_II += current_pseudopatterns_II


    # Store 4.1-specific material:
    tar_patts = []
    for p in original_training_patterns[:5*training_set_size]:
        tar_patts.append(p[1])

    custom_name = "train_set_size_" + str(training_set_size) + "_exp_1" + "turnover_rate_" + str(hpc._turnover_rate) + \
                  "weighting_" + str(hpc._weighting_dg)
    Tools.save_experiment_4_1_results(hpc, all_rand_ins, chaotically_recalled_patterns, tar_patts, custom_name,
                                      training_set_size)

    # Attempt to recall using the entire DNMM:
    sum_corr = 0.
    corr_ctr = 0.
    neocortically_recalled_pairs = []
    for [target_in, target_out] in original_training_patterns:
        obtained_in, obtained_out = ann.get_IO(target_in)
        sum_corr += Tools.get_pattern_correlation(target_out, obtained_out)
        corr_ctr += 1
        neocortically_recalled_pairs.append([obtained_in, obtained_out])
    g = sum_corr / corr_ctr

    goodness_str = "goodness of fit, g=" + "{:6.4f}".format(g)
    print goodness_str
    Tools.append_line_to_log(goodness_str)

    return [pseudopatterns_I, pseudopatterns_II, neocortically_recalled_pairs, ann, g]


def experiment_4_2_neo_version(ann, training_set_size, original_training_patterns):
    Tools.append_line_to_log("INIT. EXPERIMENT #" + str(Tools.get_experiment_counter()) + ". Type: NEO ver. 4.2.")

    pseudo_set_size = 20
    pseudopatterns_I = []
    pseudopatterns_II = []

    ann.train(original_training_patterns[:training_set_size])

    for train_set_num in range(1, 5):  # always five training sets
        start_index = training_set_size*train_set_num
        current_training_set = original_training_patterns[start_index: start_index + training_set_size]

        current_p_I = []
        current_p_II = []

        for i in range(pseudo_set_size):
            current_p_I.append(ann.get_random_IO())

            distorted_training_in = Tools.flip_bits_f(current_training_set[i % len(current_training_set)][1], flip_P=0.5)
            # current_p_II.append(ann.get_IO(distorted_training_in))

        pseudopatterns_I += current_p_I
        pseudopatterns_II += current_p_II

        for i in range(10):
            # ann.train(current_p_I + current_p_II)
            ann.train(current_training_set)

    sum_corr = 0.
    corr_ctr = 0.
    neocortically_recalled_pairs = []
    for [target_in, target_out] in original_training_patterns:
        obtained_in, obtained_out = ann.get_IO(target_in)
        sum_corr += Tools.get_pattern_correlation(target_out, obtained_out)
        corr_ctr += 1
        neocortically_recalled_pairs.append([obtained_in, obtained_out])
    g = sum_corr / corr_ctr

    goodness_str = "goodness of fit, g=" + "{:6.4f}".format(g)
    print goodness_str
    Tools.append_line_to_log(goodness_str)

    return [pseudopatterns_I, pseudopatterns_II, neocortically_recalled_pairs, ann, g]


def experiment_4_2_neo_pseudorehearsal_with_chaotic_patterns(hpc, ann, training_set_size, original_training_patterns):
    Tools.append_line_to_log("INIT. EXPERIMENT #" + str(Tools.get_experiment_counter()) + ". Type: NEO ver. 4.2.")

    chaotically_recalled_patterns = []
    all_rand_ins = []
    all_chaotic_outs = []
    pseudo_set_size = 10
    pseudopatterns_I = []
    pseudopatterns_II = []

    for train_set_num in range(5):  # always five training sets
        current_p_I = []
        current_p_II = []

        chaotically_recalled_outputs, current_random_ins = training_and_recall_hpc_helper(
                hpc, training_set_size, train_set_num, original_training_patterns)
        all_rand_ins.append(current_random_ins)
        all_chaotic_outs.append(chaotically_recalled_outputs)

        for i in range(pseudo_set_size):
            current_p_I.append(ann.get_random_IO())

            distorted_training_in = Tools.flip_bits_f(chaotically_recalled_outputs[i % len(chaotically_recalled_outputs)],
                                                      flip_P=0.5)
            current_p_II.append(ann.get_IO(distorted_training_in))

        current_chaotic_patterns = []
        for p_ctr in range(len(current_random_ins)):
            current_chaotic_patterns.append([current_random_ins[p_ctr], chaotically_recalled_outputs[p_ctr]])

        pseudopatterns_I += current_p_I
        pseudopatterns_II += current_p_II
        chaotically_recalled_patterns += current_chaotic_patterns

        for i in range(5):
            ann.train(current_p_I + current_p_II)
            ann.train(current_chaotic_patterns)

    tar_patts = []
    for pair in original_training_patterns:
        tar_patts.append(pair[1])
    Tools.save_experiment_4_1_results(hpc, all_rand_ins, all_chaotic_outs, tar_patts,
                                      "Exp. 4_2 chaotic pattern results.", training_set_size)

    sum_corr = 0.
    corr_ctr = 0.
    neocortically_recalled_pairs = []
    for [target_in, target_out] in original_training_patterns:
        obtained_in, obtained_out = ann.get_IO(target_in)
        sum_corr += Tools.get_pattern_correlation(target_out, obtained_out)
        corr_ctr += 1
        neocortically_recalled_pairs.append([obtained_in, obtained_out])
    g = sum_corr / corr_ctr

    goodness_str = "goodness of fit, g=" + "{:6.4f}".format(g)
    print goodness_str
    Tools.append_line_to_log(goodness_str)

    return [pseudopatterns_I, pseudopatterns_II, neocortically_recalled_pairs, ann, g]


def experiment_4_2_hpc_recall_every_i_iters(hpc, ann, training_set_size, original_training_patterns, train_iters):
    # LOG:
    Tools.append_line_to_log("INIT. EXPERIMENT #" + str(Tools.get_experiment_counter()) +
                             ". Type: 4.2 Chaotic recall version" +
                             ": ASYNC-flag:" + str(hpc._ASYNC_FLAG) + ". " + str(training_set_size) + "x5. " +
                             "Turnover mode: " + str(hpc._TURNOVER_MODE) + ". Turnover rate:" +
                             str(hpc._turnover_rate) + ", DG-weighting: " + str(hpc._weighting_dg) + ".")

    chaotic_recall_patterns = []
    pseudopatterns_I = []
    pseudopatterns_II = []

    test_hpc = HPC([hpc.dims[0], hpc.dims[1], hpc.dims[2], hpc.dims[3], hpc.dims[4]],
                   hpc.connection_rate_input_ec, hpc.PP, hpc.MF,  # connection rates: (in_ec, ec_dg, dg_ca3)
                   hpc.firing_rate_ec, hpc.firing_rate_dg, hpc.firing_rate_ca3,  # firing rates: (ec, dg, ca3)
                   hpc._gamma, hpc._epsilon, hpc._nu, hpc._turnover_rate,  # gamma, epsilon, nu, turnover rate
                   hpc._k_m, hpc._k_r, hpc._a_i.get_value()[0][0], hpc._alpha, hpc._weighting_dg,  # k_m, k_r, a_i, alpha. alpha is 2 in 4.1
                   _ASYNC_FLAG=hpc._ASYNC_FLAG, _TURNOVER_MODE=hpc._TURNOVER_MODE)

    for i in range(5):
        current_training_set = original_training_patterns[training_set_size*i: training_set_size*i + training_set_size]
        HPCWrappers.learn_patterns_for_i_iters_hpc_wrapper(hpc, current_training_set, train_iters)

        # append 20 chaotically recalled patterns, takes output after 15 iters of recall
        current_chaotic_recall_patts = HPCWrappers.hpc_generate_pseudopatterns_I_recall_i_iters_wrapper(
                hpc, num_of_pseudopatterns=20, num_of_iters=15)
        chaotic_recall_patterns.append(current_chaotic_recall_patts)

        test_hpc = Tools.set_to_equal_parameters(hpc, test_hpc)
        # generate pseudopatterns:
        pseudopatterns_I.append(HPCWrappers.hpc_generate_pseudopatterns_I_recall_i_iters_wrapper(
                test_hpc, num_of_pseudopatterns=20, num_of_iters=15))
        pseudopatterns_II.append(HPCWrappers.hpc_generate_pseudopatterns_II_recall_i_iters_wrapper(
                test_hpc, num_of_pseudopatterns=20, chaotically_recalled_patterns=current_chaotic_recall_patts,
                flip_P=0.5))

    # store experiment results for use in different neo. consolidation experiments
    Tools.save_chaotic_recall_results(chaotic_recall_patterns, pseudopatterns_I, pseudopatterns_II,
                                      original_training_patterns)


def experiment_4_2_hpc_recall_every_i_iters_global_exposure(hpc, ann, training_set_size, original_training_patterns, train_iters):
    # LOG:
    Tools.append_line_to_log("INIT. EXPERIMENT #" + str(Tools.get_experiment_counter()) +
                             ". Type: 4.2 Chaotic recall version" +
                             ": ASYNC-flag:" + str(hpc._ASYNC_FLAG) + ". " + str(training_set_size) + "x5. " +
                             "Turnover mode: " + str(hpc._TURNOVER_MODE) + ". Turnover rate:" +
                             str(hpc._turnover_rate) + ", DG-weighting: " + str(hpc._weighting_dg) + ".")

    chaotic_recall_patterns = []
    pseudopatterns_I = []
    pseudopatterns_II = []

    test_hpc = HPC([hpc.dims[0], hpc.dims[1], hpc.dims[2], hpc.dims[3], hpc.dims[4]],
                   hpc.connection_rate_input_ec, hpc.PP, hpc.MF,  # connection rates: (in_ec, ec_dg, dg_ca3)
                   hpc.firing_rate_ec, hpc.firing_rate_dg, hpc.firing_rate_ca3,  # firing rates: (ec, dg, ca3)
                   hpc._gamma, hpc._epsilon, hpc._nu, hpc._turnover_rate,  # gamma, epsilon, nu, turnover rate
                   hpc._k_m, hpc._k_r, hpc._a_i.get_value()[0][0], hpc._alpha, hpc._weighting_dg,  # k_m, k_r, a_i, alpha. alpha is 2 in 4.1
                   _ASYNC_FLAG=hpc._ASYNC_FLAG, _TURNOVER_MODE=hpc._TURNOVER_MODE)

    for i in range(train_iters):
        current_training_set = original_training_patterns
        HPCWrappers.learn_patterns_for_i_iters_hpc_wrapper(hpc, current_training_set, 1)

        # append 20 chaotically recalled patterns, takes output after 15 iters of recall
        current_chaotic_recall_patts = HPCWrappers.hpc_generate_pseudopatterns_I_recall_i_iters_wrapper(hpc, 10, 15)
        chaotic_recall_patterns.append(current_chaotic_recall_patts)

        test_hpc = Tools.set_to_equal_parameters(hpc, test_hpc)
        # generate pseudopatterns:
        pseudopatterns_I.append(HPCWrappers.hpc_generate_pseudopatterns_I_recall_i_iters_wrapper(test_hpc, 10, 1))
        pseudopatterns_II.append(HPCWrappers.hpc_generate_pseudopatterns_II_recall_i_iters_wrapper(
                test_hpc, num_of_pseudopatterns=20, chaotically_recalled_patterns=current_chaotic_recall_patts,
                flip_P=0.5))

    # store experiment results for use in different neo. consolidation experiments
    Tools.save_chaotic_recall_results(chaotic_recall_patterns, pseudopatterns_I, pseudopatterns_II,
                                      original_training_patterns)


# ========================== RANDOM STREAMS IN ==================================
def experiment_4_2_hpc_recall_every_i_iters_random_stream(hpc, ann, training_set_size, original_training_patterns, train_iters):
    # LOG:
    Tools.append_line_to_log("INIT. EXPERIMENT #" + str(Tools.get_experiment_counter()) +
                             ". Type: 4.2 Chaotic recall version" +
                             ": ASYNC-flag:" + str(hpc._ASYNC_FLAG) + ". " + str(training_set_size) + "x5. " +
                             "Turnover mode: " + str(hpc._TURNOVER_MODE) + ". Turnover rate:" +
                             str(hpc._turnover_rate) + ", DG-weighting: " + str(hpc._weighting_dg) + ".")

    chaotic_recall_patterns = []
    pseudopatterns_I = []
    pseudopatterns_II = []

    test_hpc = HPC([hpc.dims[0], hpc.dims[1], hpc.dims[2], hpc.dims[3], hpc.dims[4]],
                   hpc.connection_rate_input_ec, hpc.PP, hpc.MF,  # connection rates: (in_ec, ec_dg, dg_ca3)
                   hpc.firing_rate_ec, hpc.firing_rate_dg, hpc.firing_rate_ca3,  # firing rates: (ec, dg, ca3)
                   hpc._gamma, hpc._epsilon, hpc._nu, hpc._turnover_rate,  # gamma, epsilon, nu, turnover rate
                   hpc._k_m, hpc._k_r, hpc._a_i.get_value()[0][0], hpc._alpha, hpc._weighting_dg,  # k_m, k_r, a_i, alpha. alpha is 2 in 4.1
                   _ASYNC_FLAG=hpc._ASYNC_FLAG, _TURNOVER_MODE=hpc._TURNOVER_MODE)

    for i in range(5):
        current_training_set = original_training_patterns[training_set_size*i: training_set_size*i + training_set_size]
        HPCWrappers.learn_patterns_for_i_iters_hpc_wrapper(hpc, current_training_set, train_iters)

        # append 20 chaotically recalled patterns, takes output after 15 iters of recall
        current_chaotic_recall_patts = HPCWrappers.hpc_generate_pseudopatterns_I_recall_i_iters_wrapper_random_stream(
                hpc, num_of_pseudopatterns=20, num_of_iters=15)
        chaotic_recall_patterns.append(current_chaotic_recall_patts)

        test_hpc = Tools.set_to_equal_parameters(hpc, test_hpc)
        # generate pseudopatterns:
        pseudopatterns_I.append(HPCWrappers.hpc_generate_pseudopatterns_I_recall_i_iters_wrapper_random_stream(
                test_hpc, num_of_pseudopatterns=20, num_of_iters=15))
        pseudopatterns_II.append(HPCWrappers.hpc_generate_pseudopatterns_II_recall_i_iters_wrapper(
                test_hpc, num_of_pseudopatterns=20, chaotically_recalled_patterns=current_chaotic_recall_patts,
                flip_P=0.5))

    # store experiment results for use in different neo. consolidation experiments
    Tools.save_chaotic_recall_results(chaotic_recall_patterns, pseudopatterns_I, pseudopatterns_II,
                                      original_training_patterns)


def experiment_4_2_hpc_recall_every_i_iters_global_exposure_random_stream(hpc, ann, training_set_size, original_training_patterns, train_iters):
    # LOG:
    Tools.append_line_to_log("INIT. EXPERIMENT #" + str(Tools.get_experiment_counter()) +
                             ". Type: 4.2 Chaotic recall version" +
                             ": ASYNC-flag:" + str(hpc._ASYNC_FLAG) + ". " + str(training_set_size) + "x5. " +
                             "Turnover mode: " + str(hpc._TURNOVER_MODE) + ". Turnover rate:" +
                             str(hpc._turnover_rate) + ", DG-weighting: " + str(hpc._weighting_dg) + ".")

    chaotic_recall_patterns = []
    pseudopatterns_I = []
    pseudopatterns_II = []

    test_hpc = HPC([hpc.dims[0], hpc.dims[1], hpc.dims[2], hpc.dims[3], hpc.dims[4]],
                   hpc.connection_rate_input_ec, hpc.PP, hpc.MF,  # connection rates: (in_ec, ec_dg, dg_ca3)
                   hpc.firing_rate_ec, hpc.firing_rate_dg, hpc.firing_rate_ca3,  # firing rates: (ec, dg, ca3)
                   hpc._gamma, hpc._epsilon, hpc._nu, hpc._turnover_rate,  # gamma, epsilon, nu, turnover rate
                   hpc._k_m, hpc._k_r, hpc._a_i.get_value()[0][0], hpc._alpha, hpc._weighting_dg,  # k_m, k_r, a_i, alpha. alpha is 2 in 4.1
                   _ASYNC_FLAG=hpc._ASYNC_FLAG, _TURNOVER_MODE=hpc._TURNOVER_MODE)

    for i in range(train_iters):
        current_training_set = original_training_patterns
        HPCWrappers.learn_patterns_for_i_iters_hpc_wrapper(hpc, current_training_set, 1)

        # append 20 chaotically recalled patterns, takes output after 15 iters of recall
        current_chaotic_recall_patts = HPCWrappers.hpc_generate_pseudopatterns_I_recall_i_iters_wrapper_random_stream(hpc, 10, 15)
        chaotic_recall_patterns.append(current_chaotic_recall_patts)

        test_hpc = Tools.set_to_equal_parameters(hpc, test_hpc)
        # generate pseudopatterns:
        pseudopatterns_I.append(HPCWrappers.hpc_generate_pseudopatterns_I_recall_i_iters_wrapper_random_stream(test_hpc, 10, 1))
        pseudopatterns_II.append(HPCWrappers.hpc_generate_pseudopatterns_II_recall_i_iters_wrapper(
                test_hpc, num_of_pseudopatterns=20, chaotically_recalled_patterns=current_chaotic_recall_patts,
                flip_P=0.5))

    # store experiment results for use in different neo. consolidation experiments
    Tools.save_chaotic_recall_results(chaotic_recall_patterns, pseudopatterns_I, pseudopatterns_II,
                                      original_training_patterns)


def experiment_4_2_hpc_generate_output_images_for_every_learning_iteration(hpc, ann, training_set_size, original_training_patterns, train_iters):
    # LOG:
    Tools.append_line_to_log("INIT. EXPERIMENT #" + str(Tools.get_experiment_counter()) +
                             ". Type: 4.2 Chaotic recall version" +
                             ": ASYNC-flag:" + str(hpc._ASYNC_FLAG) + ". " + str(training_set_size) + "x5. " +
                             "Turnover mode: " + str(hpc._TURNOVER_MODE) + ". Turnover rate:" +
                             str(hpc._turnover_rate) + ", DG-weighting: " + str(hpc._weighting_dg) + ".")

    chaotic_recall_patterns = []
    pseudopatterns_I = []
    pseudopatterns_II = []

    test_hpc = HPC([hpc.dims[0], hpc.dims[1], hpc.dims[2], hpc.dims[3], hpc.dims[4]],
                   hpc.connection_rate_input_ec, hpc.PP, hpc.MF,  # connection rates: (in_ec, ec_dg, dg_ca3)
                   hpc.firing_rate_ec, hpc.firing_rate_dg, hpc.firing_rate_ca3,  # firing rates: (ec, dg, ca3)
                   hpc._gamma, hpc._epsilon, hpc._nu, hpc._turnover_rate,  # gamma, epsilon, nu, turnover rate
                   hpc._k_m, hpc._k_r, hpc._a_i.get_value()[0][0], hpc._alpha, hpc._weighting_dg,  # k_m, k_r, a_i, alpha. alpha is 2 in 4.1
                   _ASYNC_FLAG=hpc._ASYNC_FLAG, _TURNOVER_MODE=hpc._TURNOVER_MODE)

    io_trials = []
    for i in range(train_iters):
        current_training_set = original_training_patterns
        HPCWrappers.learn_patterns_for_i_iters_hpc_wrapper(hpc, current_training_set, 1)

        # append 20 chaotically recalled patterns, takes output after 15 iters of recall
        current_chaotic_recall_patts = HPCWrappers.hpc_generate_pseudopatterns_I_recall_i_iters_wrapper(hpc, 10, 15)
        chaotic_recall_patterns.append(current_chaotic_recall_patts)

        test_hpc = Tools.set_to_equal_parameters(hpc, test_hpc)
        current_io_trials = Tools.generate_recall_attempt_results(test_hpc, current_training_set)
        io_trials.append(current_io_trials)
        # generate pseudopatterns:
        pseudopatterns_I.append(HPCWrappers.hpc_generate_pseudopatterns_I_recall_i_iters_wrapper(test_hpc, 10, 1))
        pseudopatterns_II.append(HPCWrappers.hpc_generate_pseudopatterns_II_recall_i_iters_wrapper(
                test_hpc, num_of_pseudopatterns=20, chaotically_recalled_patterns=current_chaotic_recall_patts,
                flip_P=0.5))

    io_ctr = 0
    for io_trial in io_trials:
        Tools.save_aggregate_image_from_IOs(io_trial, 'recall_trial', io_ctr)
        io_ctr += 1

    chaotic_ctr = 0
    for chaotically_recalled_pattern_set in chaotic_recall_patterns:
        Tools.save_aggregate_image_from_IOs(chaotically_recalled_pattern_set, 'chaotically_recalled', chaotic_ctr)
        chaotic_ctr += 1

    # store experiment results for use in different neo. consolidation experiments
    Tools.save_chaotic_recall_results(chaotic_recall_patterns, pseudopatterns_I, pseudopatterns_II,
                                      original_training_patterns)
