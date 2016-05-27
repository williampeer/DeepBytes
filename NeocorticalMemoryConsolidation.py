from NeocorticalNetwork import NeocorticalNetwork
import Tools
from DataWrapper import training_patterns_associative as training_patterns


def evaluate_goodness_of_fit(ann, target_patterns):
    sum_g = 0.

    assert ann.classType is type(NeocorticalNetwork)

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

    for exp_index in range(start_index, stop_index+1):
        current_chaotic_patterns, current_pseudopatterns = \
            Tools.retrieve_patterns_for_consolidation(exp_index, exp_index%4 + 2)  # 2-5 looped
        training_set = get_training_set_from_patterns_in_scheme(current_chaotic_patterns, current_pseudopatterns,
                                                                scheme_num)
        ann15 = get_ann_trained_on_patterns(training_patterns=training_set, training_iterations=15)
        ann1k = get_ann_trained_on_patterns(training_patterns=training_set, training_iterations=1000)

        Tools.append_line_to_log('Neocortical module consolidation. Scheme: #'+str(scheme_num)+'Exp#'+str(exp_index)+
                                 '\n15 iters: g='+str(
            evaluate_goodness_of_fit(ann15, get_target_patterns(exp_index%4+2)))+
                                 '\n1k iters: g=' + str(
            evaluate_goodness_of_fit(ann1k, get_target_patterns(exp_index % 4 + 2))))


def get_target_patterns(subset_size):
    return training_patterns[:5*subset_size]


def get_training_set_from_patterns_in_scheme(chaotic_patterns, pseudopatterns, scheme):
    if scheme==0:
        return unwrapper(chaotic_patterns)
    elif scheme==1:
        train_set = []
        for s_ctr in range(len(chaotic_patterns)):
            train_set += chaotic_patterns[s_ctr] + pseudopatterns[0][s_ctr] + pseudopatterns[1][s_ctr]
        return train_set
    elif scheme==2:
        train_set = []
        for s_ctr in range(len(chaotic_patterns)):
            train_set += chaotic_patterns[s_ctr] + pseudopatterns[0][s_ctr]
    elif scheme == 3:
        train_set = []
        for s_ctr in range(len(chaotic_patterns)):
            train_set += chaotic_patterns[s_ctr] + pseudopatterns[1][s_ctr]


def unwrapper(patterns):
    unwrapped_patterns = []
    for set in patterns:
        unwrapped_patterns += set
    return unwrapped_patterns


iterate_over_experiments_suite(720, 721, 0)
