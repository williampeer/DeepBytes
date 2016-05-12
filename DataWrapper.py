import numpy as np
from data_capital import data_letters_capital
from data_lowercase import data_letters_lowercase

l_ctr = 0
for letter in data_letters_lowercase:
    if len(letter) != 7:
        print "not 7 rows, letter #", l_ctr
    row_ctr = 0
    for row in letter:
        if len(row) != 7:
            print "row not 7 els, letter #", l_ctr, "row #", row_ctr
        row_ctr += 1
    l_ctr += 1

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
    uppercase_letter = training_patterns_associative[letter_ctr % len(training_patterns_associative)][0]
    letter_ctr += 1
    training_patterns_heterogeneous.append([uppercase_letter, lowercase_letter])
