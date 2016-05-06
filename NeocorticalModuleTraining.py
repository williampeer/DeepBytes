import Tools
import cPickle
from NeocorticalNetwork import NeocorticalNetwork
from DataWrapper import training_patterns_associative
# from DataWrapper import training_patterns_heterogeneous


def traditional_training():
    io_dim = 49
    ann = NeocorticalNetwork(io_dim, 30, io_dim, 0.01, 0.9)
    training_set = training_patterns_associative[:25]
    ss = 2
    for i in range(5):
        for training_iterations in range(5):
            ann.train(training_set[i*ss:i*ss+ss])
    # for j in range(ss*5):
    #     Tools.show_image_from(ann.get_IO(training_set[j][0])[1])

    return ann


def global_sequential_FFBP_training():
    io_dim = 49
    ann = NeocorticalNetwork(io_dim, 30, io_dim, 0.01, 0.9)
    ss = 2
    training_set = training_patterns_associative[:5*ss]
    for i in range(10):
        ann.train(training_set)

    return ann


def retrieve_chaotic_patterns_from_exp_num(exp_num):
    prefix = 'saved_data/chaotic_pattern_recalls_set_size_2/'
    chaotic_out_filename = '_chaotically_recalled_patterns_exp#' + str(exp_num) + '.save'
    rand_in_filename = '_corresponding_random_ins_exp#' + str(exp_num) + '.save'

    chaotic_out_file = file(prefix+chaotic_out_filename, 'rb')
    chaotic_out = cPickle.load(chaotic_out_file)
    chaotic_out_file.close()

    rand_in_file = file(prefix+rand_in_filename, 'rb')
    rand_ins = cPickle.load(rand_in_file)
    rand_in_file.close()

    return [chaotic_out, rand_ins]


def train_on_chaotic_patterns():
    chaotic_outs, rand_ins = retrieve_chaotic_patterns_from_exp_num(515)
    chaotic_patts = []
    for i in range(len(chaotic_outs)):
        chaotic_patts.append([])
        for j in range(len(chaotic_outs[0])):
            chaotic_patts[i].append([chaotic_outs[i][j], rand_ins[i][j]])

    io_dim = 49
    ann = NeocorticalNetwork(io_dim, 30, io_dim, 0.01, 0.9)
    training_set = chaotic_patts[0]  # first two
    for training_iterations in range(15):
        ann.train(training_set)
        for j in range(2):
            Tools.show_image_from(ann.get_IO(training_patterns_associative[j][0])[1])
# Tools.show_image_from(training_patterns_associative[15][0])
# traditional_training()
# train_on_chaotic_patterns()
