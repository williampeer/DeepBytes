import numpy as np

io_dim = 49

# sample IO:
I = np.asarray([[1, -1, 1, -1, 1, -1, 1] * 7], dtype=np.float32)
O = np.asarray([[-1, 1, -1, 1, -1, 1, -1] * 7], dtype=np.float32)

rand_I = np.random.random((1, io_dim)).astype(np.float32) - 0.5 * np.ones((1, io_dim), dtype=np.float32)
rand_O = np.random.random((1, io_dim)).astype(np.float32) - 0.5 * np.ones((1, io_dim), dtype=np.float32)
for index in xrange(rand_I.shape[1]):
    if rand_I[0][index] < 0:
        rand_I[0][index] = -1
    else:
        rand_I[0][index] = 1
    if rand_O[0][index] < 0:
        rand_O[0][index] = -1
    else:
        rand_O[0][index] = 1