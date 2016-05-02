import numpy as np
from data_capital import data_letters_capital
from data_lowercase import data_letters_lowercase

training_patterns_associative = []
# Setup all training patterns:
for letter_data in data_letters_capital:
    io = [[]]
    for row in letter_data:
        for el in row:
            io[0].append(el)
    new_array = np.asarray(io, dtype=np.float32)
    training_patterns_associative.append([new_array, new_array])

training_patterns_heterogeneous = []
letter_ctr = 0
for letter_data in data_letters_lowercase:
    io_lowercase = [[]]
    for row in letter_data:
        for el in row:
            io_lowercase[0].append(el)
    lowercase_letter = np.asarray(io_lowercase, dtype=np.float32)
    uppercase_letter = training_patterns_associative[letter_ctr][0]
    training_patterns_heterogeneous.append([uppercase_letter, lowercase_letter])