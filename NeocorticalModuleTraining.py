import Tools
import cPickle
from NeocorticalNetwork import NeocorticalNetwork
from DataWrapper import training_patterns_associative
# from DataWrapper import training_patterns_heterogeneous


def traditional_training_with_catastrophic_interference(ss):
    io_dim = 49
    ann = NeocorticalNetwork(io_dim, 30, io_dim, 0.01, 0.9)
    training_set = training_patterns_associative[:25]
    # ss = 2
    for i in range(5):
        for training_iterations in range(15):
            ann.train(training_set[i*ss:i*ss+ss])
    # for j in range(ss*5):
    #     Tools.show_image_from(ann.get_IO(training_set[j][0])[1])

    return ann


def global_sequential_FFBP_training(ss):
    io_dim = 49
    ann = NeocorticalNetwork(io_dim, 30, io_dim, 0.01, 0.9)
    training_set = training_patterns_associative[:5*ss]
    for i in range(15):
        ann.train(training_set)

    return ann


def retrieve_chaotic_patterns_from_exp_num(exp_num):
    prefix = 'saved_data/chaotic_pattern_recalls_set_size_2/'
    chaotic_out_filename = '_chaotically_recalled_patterns_exp#' + str(exp_num) + '.save'
    rand_in_filename = '_corresponding_random_ins_exp#' + str(exp_num) + '.save'

    chaotic_out_file = file(prefix+chaotic_out_filename, 'rb')
    chaotic_out = cPickle.load(chaotic_out_file)
    chaotic_out_file.close()

    rand_in_file = file(prefix + rand_in_filename, 'rb')
    rand_ins = cPickle.load(rand_in_file)
    rand_in_file.close()

    return [chaotic_out, rand_ins]


def train_on_chaotic_patterns():
    chaotic_outs, rand_ins = retrieve_chaotic_patterns_from_exp_num(1)
    # print "len(chaotic_outs):", len(chaotic_outs)
    chaotic_patts = []
    for i in range(len(chaotic_outs)):
        chaotic_patts.append([])
        for j in range(len(chaotic_outs[i])):
            chaotic_patts[i].append([rand_ins[i][j], chaotic_outs[i][j]])

    io_dim = 49
    ann = NeocorticalNetwork(io_dim, 30, io_dim, 0.01, 0.9)

    training_set = []
    for i in range(5):
        training_set += chaotic_patts[i]
        ann.train(chaotic_patts[i])
    for train_iters in range(10):
        pass

    return ann


def evaluate_ann(ann, set_size):
    print "Evaluating the ANN-object.."
    sum_corr = 0.
    corr_ctr = 0.
    neocortically_recalled_pairs = []
    for [target_in, target_out] in training_patterns_associative[:5*set_size]:
        obtained_in, obtained_out = ann.get_IO(target_in)
        sum_corr += Tools.get_pattern_correlation(target_out, obtained_out)
        corr_ctr += 1
        neocortically_recalled_pairs.append([obtained_in, obtained_out])
    g = sum_corr / corr_ctr

    goodness_str = "goodness of fit, g=" + "{:6.4f}".format(g)
    print goodness_str
    return g


def evaluate_ann_with_bipolar_output(ann, set_size):
    print "Evaluating the ANN-object.."
    sum_corr = 0.
    corr_ctr = 0.
    neocortically_recalled_pairs = []
    for [target_in, target_out] in training_patterns_associative[:5*set_size]:
        obtained_in, obtained_out = ann.get_IO(target_in)
        sum_corr += Tools.get_pattern_correlation(target_out, Tools.get_bipolar_in_out_values(obtained_out))
        corr_ctr += 1
        neocortically_recalled_pairs.append([obtained_in, obtained_out])
    g = sum_corr / corr_ctr

    goodness_str = "goodness of fit, g=" + "{:6.4f}".format(g)
    print goodness_str

# Tools.show_image_from(training_patterns_associative[15][0])
# goodness_values = []
# for i in range(20):
#     exp_results = []
#     for ss in range(2, 6):
#         ann = traditional_training_with_catastrophic_interference(ss)
#         exp_results.append(evaluate_ann(ann, ss))
#     goodness_values.append(exp_results)
#
# print "goodness_values:", goodness_values

# ann = global_sequential_FFBP_training()
# ann = train_on_chaotic_patterns()
# evaluate_ann_with_bipolar_output(ann, ss)
