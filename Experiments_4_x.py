import time, HPCWrappers, NeocorticalModuleTraining
from HPCWrappers import hpc_learn_patterns_wrapper, hpc_chaotic_recall_wrapper
from Tools import set_contains_pattern, get_pattern_correlation, save_experiment_4_1_results, save_images_from
import Tools
import numpy as np

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
        sum_corr += get_pattern_correlation(target_out, obtained_out)
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
        sum_corr += get_pattern_correlation(target_out, obtained_out)
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
        all_rand_ins += current_random_ins
        all_chaotic_outs += chaotically_recalled_outputs

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

        # for i in range(5):
        ann.train(current_chaotic_patterns)
        # ann.train(current_p_I + current_p_II)

    Tools.save_experiment_4_1_results(hpc, all_rand_ins, all_chaotic_outs, original_training_patterns,
                                      "Exp. 4_2 chaotic pattern results.", training_set_size)

    sum_corr = 0.
    corr_ctr = 0.
    neocortically_recalled_pairs = []
    for [target_in, target_out] in original_training_patterns:
        obtained_in, obtained_out = ann.get_IO(target_in)
        sum_corr += get_pattern_correlation(target_out, obtained_out)
        corr_ctr += 1
        neocortically_recalled_pairs.append([obtained_in, obtained_out])
    g = sum_corr / corr_ctr

    goodness_str = "goodness of fit, g=" + "{:6.4f}".format(g)
    print goodness_str
    Tools.append_line_to_log(goodness_str)

    return [pseudopatterns_I, pseudopatterns_II, neocortically_recalled_pairs, ann, g]


# def experiment_4_2_hpc_version(hpc, ann, training_set_size, original_training_patterns):
#     Tools.append_line_to_log("INIT. EXPERIMENT #" + str(Tools.get_experiment_counter()) + ". Type: HPC ver. 4.2" +
#                              ": ASYNC-flag:" + str(hpc._ASYNC_FLAG) + ". " + str(training_set_size) + "x5. " +
#                              "Turnover mode: " + str(hpc._TURNOVER_MODE) + ". Turnover rate:" +
#                              str(hpc._turnover_rate) + ", DG-weighting: " + str(hpc._weighting_dg) + ".")
#     #
#     # chaotically_recalled_patterns = []
#     # all_rand_ins = []
#     #
#     # pseudopatterns_I = []
#     # pseudopatterns_II = []
#
#     for train_set_num in range(5):  # always five training sets
#     #     current_set_hipp_chaotic_recall, current_set_random_ins = \
#     #         training_and_recall_hpc_helper(hpc, training_set_size, train_set_num, original_training_patterns)
#
#     #     chaotic_training_set = []
#     #     for p_ctr in range(len(current_set_hipp_chaotic_recall)):
#     #         # if not Tools.set_contains_pattern(chaotically_recalled_patterns, current_set_hipp_chaotic_recall[p_ctr]):
#     #         chaotically_recalled_patterns.append([current_set_hipp_chaotic_recall[p_ctr]])
#     #         all_rand_ins.append([current_set_random_ins[p_ctr]])
#     #         chaotic_training_set.append([current_set_random_ins[p_ctr], current_set_hipp_chaotic_recall[p_ctr]])
#
#         for i in range(5):
#             # ann.train(chaotic_training_set)
#             start_index = train_set_num*training_set_size
#             ann.train(original_training_patterns[start_index: start_index + training_set_size])
#
#     # ann = NeocorticalModuleTraining.global_sequential_FFBP_training()
#
#     # Store 4.1-specific material:
#     tar_patts = []
#     for p in original_training_patterns[:5*training_set_size]:
#         tar_patts.append(p[1])
#
#     custom_name = "train_set_size_" + str(training_set_size) + "_exp_1" + "turnover_rate_" + str(hpc._turnover_rate) + \
#                   "weighting_" + str(hpc._weighting_dg)
#     Tools.save_experiment_4_1_results(hpc, all_rand_ins, chaotically_recalled_patterns, tar_patts, custom_name,
#                                       training_set_size)
#
#     # Attempt to recall using the entire DNMM:
#     sum_corr = 0.
#     corr_ctr = 0.
#     neocortically_recalled_pairs = []
#     for [target_in, target_out] in original_training_patterns:
#         obtained_in, obtained_out = ann.get_IO(target_in)
#         # obtained_out = Tools.get_bipolar_in_out_values(obtained_out)
#         # identical = True
#         # for i in range(len(obtained_out)):
#         #     for j in range(len(obtained_out[0])):
#         #         if obtained_out[i][j] != target_out[i][j]:
#         #             print "not identical, oo[i][j]:", obtained_out[i][j], ", taro[i][j]:", target_out[i][j]
#         #             identical = False
#         #             break
#         # if identical:
#         #     if get_pattern_correlation(target_out, obtained_out) != 1.0:
#         #         print "patterns are identical, but corr is:", get_pattern_correlation(target_out, obtained_out)
#         # else:
#         #     print "current corr:", get_pattern_correlation(target_out, obtained_out)
#         sum_corr += get_pattern_correlation(target_out, obtained_out)
#         corr_ctr += 1
#         neocortically_recalled_pairs.append([obtained_in, obtained_out])
#     g = sum_corr / corr_ctr
#
#     goodness_str = "goodness of fit, g=" + "{:6.4f}".format(g)
#     print goodness_str
#     Tools.append_line_to_log(goodness_str)
#
#     return [pseudopatterns_I, pseudopatterns_II, neocortically_recalled_pairs, ann, g]
