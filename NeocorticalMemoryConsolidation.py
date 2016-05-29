from NeocorticalNetwork import NeocorticalNetwork
import Tools
from DataWrapper import training_patterns_associative as training_patterns
import time


def evaluate_goodness_of_fit(ann, target_patterns):
    sum_g = 0.

    assert isinstance(ann, NeocorticalNetwork)

    # recall
    for pattern in target_patterns:
        _, output = ann.get_IO(pattern[0])
        sum_g += Tools.get_pattern_correlation(pattern[1], output)

    return sum_g / float(len(target_patterns))  # returns the goodness of fit


def get_ann_trained_on_patterns(training_patterns, training_iterations):
    ann = NeocorticalNetwork(49, 30, 49, 0.01, 0.9)

    # training:
    for x in xrange(training_iterations):
        ann.train(training_patterns)

    return ann


def iterate_over_experiments_suite(start_index, stop_index, scheme_num):
    ann = NeocorticalNetwork(49, 30, 49, 0.01, 0.9)

    for exp_index in range(start_index, stop_index):
        current_chaotic_patterns, current_pseudopatterns = \
            Tools.retrieve_patterns_for_consolidation(exp_index, exp_index%4 + 2)  # 2-5 looped
        training_set = get_training_set_from_patterns_in_scheme_full_set(current_chaotic_patterns,
                                                                         current_pseudopatterns, scheme_num)
        t0 = time.time()
        ann.reset()
        for i in range(15):
            ann.train(training_set)
        results_line = 'Neocortical module consolidation. Scheme: '+str(scheme_num)+'. Exp#'+str(exp_index)+ \
                       '\n15 iters: g='+str(evaluate_goodness_of_fit(ann, get_target_patterns(exp_index%4+2)))

        ann.reset()
        for i in range(200):
            ann.train(training_set)
        results_line += '\n215 iters: g=' + str(evaluate_goodness_of_fit(ann, get_target_patterns(exp_index % 4 + 2)))
        t1 = time.time()
        print 'Trained and evaluated performance in '+'{:8.3f}'.format(t1-t0), 'seconds'
        print results_line
        Tools.append_line_to_log(results_line)


def iterate_over_experiments_suite_halved_pseudopattern_size(start_index, stop_index, scheme_num):

    for exp_index in range(start_index, stop_index):
        current_chaotic_patterns, current_pseudopatterns = \
            Tools.retrieve_patterns_for_consolidation(exp_index, exp_index%4 + 2)  # 2-5 looped
        training_set = get_training_set_from_patterns_in_scheme_half_pseudopatterns(current_chaotic_patterns,
                                                                                    current_pseudopatterns, scheme_num)

        t0 = time.time()
        ann = get_ann_trained_on_patterns(training_patterns=training_set, training_iterations=15)
        results_line = 'Neocortical module consolidation. Halved pseudopattern set size. Scheme: '+str(scheme_num)+\
                       '. Exp#'+str(exp_index)+'\n15 iters: g='+\
                       str(evaluate_goodness_of_fit(ann, get_target_patterns(exp_index%4+2)))

        for i in range(200):
            ann.train(training_set)
        results_line += '\n1k iters: g=' + str(evaluate_goodness_of_fit(ann, get_target_patterns(exp_index % 4 + 2)))
        t1 = time.time()
        print 'Trained and evaluated performance in '+'{:8.3f}'.format(t1-t0), 'seconds'
        print results_line
        Tools.append_line_to_log(results_line)


def iterate_over_experiments_suite_span_output_demo(start_index, stop_index):
    ann = NeocorticalNetwork(49, 30, 49, 0.01, 0.9)

    for exp_index in range(start_index, stop_index):
        current_chaotic_patterns, current_pseudopatterns = \
            Tools.retrieve_patterns_for_consolidation(exp_index, exp_index%4 + 2)  # 2-5 looped
        training_set = []

        t0 = time.time()
        ann.reset()
        for cp_subset in current_chaotic_patterns:
            training_subset = []
            for cp in cp_subset:
                training_subset.append([cp[1], cp[1]])
            for i in range(15):
                ann.train(training_subset)

        results_line = 'Neocortical module consolidation. Output as IO. Exp#'+str(exp_index)+\
                       '\n'+str(i+1)+' iters: g='+str(evaluate_goodness_of_fit(ann, get_target_patterns(exp_index%4+2)))

        ann.reset()
        for cp_subset in current_chaotic_patterns:
            training_subset = []
            for cp in cp_subset:
                training_subset.append([cp[1], cp[1]])
            for i in range(200):
                ann.train(training_subset)

        results_line += '\n'+str(i+1)+' iters: g=' + str(evaluate_goodness_of_fit(ann, get_target_patterns(exp_index % 4 + 2)))
        t1 = time.time()
        print 'Trained and evaluated performance in '+'{:8.3f}'.format(t1-t0), 'seconds'
        print results_line
        Tools.append_line_to_log(results_line)


def iterate_over_experiments_suite_span_output_demo_global(start_index, stop_index):
    ann = NeocorticalNetwork(49, 30, 49, 0.01, 0.9)

    for exp_index in range(start_index, stop_index):
        current_chaotic_patterns, current_pseudopatterns = \
            Tools.retrieve_patterns_for_consolidation(exp_index, exp_index%4 + 2)  # 2-5 looped
        training_set = []
        for cp_subset in current_chaotic_patterns:
            for cp in cp_subset:
                training_set.append([cp[1], cp[1]])

        t0 = time.time()
        ann.reset()
        for i in range(15):
            ann.train(training_set)
        results_line = 'Neocortical module consolidation. Output as IO. Exp#'+str(exp_index)+\
                       '\n'+str(i+1)+' iters: g='+str(evaluate_goodness_of_fit(ann, get_target_patterns(exp_index%4+2)))

        ann.reset()
        for i in range(200):
            ann.train(training_set)
        results_line += '\n'+str(i+1)+' iters: g=' + str(evaluate_goodness_of_fit(ann, get_target_patterns(exp_index % 4 + 2)))
        t1 = time.time()
        print 'Trained and evaluated performance in '+'{:8.3f}'.format(t1-t0), 'seconds'
        print results_line
        Tools.append_line_to_log(results_line)


def get_target_patterns(subset_size):
    return training_patterns[:5*subset_size]


# for each subset in i iters: 20 chaotically recalled patterns, 10 pseudopatterns of type i, the same of type ii
def get_training_set_from_patterns_in_scheme_full_set(chaotic_patterns, pseudopatterns, scheme):
    train_set = []
    if scheme==0:
        return unwrapper(chaotic_patterns)
    elif scheme==1:
        for s_ctr in range(len(chaotic_patterns)):
            train_set += chaotic_patterns[s_ctr] + pseudopatterns[0][s_ctr] + pseudopatterns[1][s_ctr]
        return train_set
    elif scheme==2:
        for s_ctr in range(len(chaotic_patterns)):
            train_set += chaotic_patterns[s_ctr] + pseudopatterns[0][s_ctr]
    elif scheme == 3:
        for s_ctr in range(len(chaotic_patterns)):
            train_set += chaotic_patterns[s_ctr] + pseudopatterns[1][s_ctr]

    return train_set


def get_training_set_from_patterns_in_scheme_half_pseudopatterns(chaotic_patterns, pseudopatterns, scheme):
    train_set = []
    if scheme == 0:
        return unwrapper(chaotic_patterns)
    elif scheme == 1:
        for s_ctr in range(len(chaotic_patterns)):
            train_set += chaotic_patterns[s_ctr] + pseudopatterns[0][s_ctr][:len(pseudopatterns[0][s_ctr]) / 2] + \
                         pseudopatterns[1][s_ctr][:len(pseudopatterns[1][s_ctr]) / 2]
        return train_set
    elif scheme == 2:
        for s_ctr in range(len(chaotic_patterns)):
            train_set += chaotic_patterns[s_ctr] + pseudopatterns[0][s_ctr][:len(pseudopatterns[0][s_ctr]) / 2]
    elif scheme == 3:
        for s_ctr in range(len(chaotic_patterns)):
            train_set += chaotic_patterns[s_ctr] + pseudopatterns[1][s_ctr][:len(pseudopatterns[1][s_ctr]) / 2]

    return train_set


def unwrapper(patterns):
    unwrapped_patterns = []
    for set in patterns:
        unwrapped_patterns += set
    return unwrapped_patterns


# perform all schemes for both suites
iterate_over_experiments_suite_span_output_demo(80, 160)
# iterate_over_experiments_suite_span_output_demo_global(80, 160)
# for scheme_ctr in range(4):
    # iterate_over_experiments_suite(80, 160, scheme_num=scheme_ctr)
    # iterate_over_experiments_suite_halved_pseudopattern_size(80, 160, scheme_num=scheme_ctr)

# test_chaotic_patterns, test_pseudopatterns = \
#     Tools.retrieve_patterns_for_consolidation(720, 720 % 4 + 2)  # 2-5 looped
# test_training_set20 = get_training_set_from_patterns_in_scheme_twenty_patterns(test_chaotic_patterns, test_pseudopatterns, 1)
# test_training_set_full = get_training_set_from_patterns_in_scheme_full_set(test_chaotic_patterns, test_pseudopatterns, 1)
# test_pi = test_pseudopatterns[0]
# test_pii = test_pseudopatterns[1]
# print len(unwrapper(test_chaotic_patterns))
# print len(unwrapper(test_pi))
# print len(unwrapper(test_pii))
# print len(test_training_set20)
# print len(test_training_set_full)
