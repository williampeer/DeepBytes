import Tools
import numpy as np
import theano
from NeocorticalNetwork import NeocorticalNetwork
from DataWrapper import training_patterns_associative
from DataWrapper import training_patterns_heterogeneous

def traditional_training():
    io_dim = 49
    ann = NeocorticalNetwork(io_dim, 30, io_dim, 0.01, 0.9)
    training_set = training_patterns_associative[:25]
    ss = 2
    for i in range(5):
        for training_iterations in range(5):
            ann.train(training_set[i*ss:i*ss+ss])
    for j in range(ss*5):
        Tools.show_image_from(ann.get_IO(training_set[j][0])[1])

# Tools.show_image_from(training_patterns_associative[15][0])
traditional_training()