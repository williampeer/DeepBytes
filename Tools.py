import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from PIL import Image
import numpy as np


def show_image_from(out_now):
    width = 7
    height = 7
    pixel_scaling_factor = 2 ** 3  # Exponent of two for symmetry.
    im = Image.new('1', (width*pixel_scaling_factor, height*pixel_scaling_factor))
    for element in range(out_now.shape[1]):
        for i in range(pixel_scaling_factor):
            for j in range(pixel_scaling_factor):
                im.putpixel(((element % width)*pixel_scaling_factor + j,
                             np.floor(element/height).astype(np.int8) * pixel_scaling_factor + i),
                            out_now[0][element]*255)
    im.show()
    # print "Output image"


shared_random_generator = RandomStreams()

x_r = T.iscalar()
y_r = T.iscalar()
p_scalar = T.fscalar('p_scalar')
binomial_f = theano.function([x_r, y_r, p_scalar], outputs=shared_random_generator.
                             binomial(size=(x_r, y_r), n=1, p=p_scalar, dtype='float32'))

rows = T.iscalar()
columns = T.iscalar()
uniform_f = theano.function([rows, columns], outputs=shared_random_generator.
                            uniform(size=(rows, columns), low=-0.1, high=0.1, dtype='float32'))


def set_contains_pattern(set, pattern):
    for pat in set:
        if get_pattern_correlation(pat, pattern) == 1:
            return True
    return False

pat1 = T.fmatrix()
pat2 = T.fmatrix()
get_pattern_correlation = theano.function([pat1, pat2], outputs=T.sum(pat1 * pat2)/(pat1.shape[0] * pat1.shape[1]))


def get_pattern_correlation_slow(pat1, pat2):
    corr = 0
    for row_ind in range(len(pat1)):
        for col_ind in range(len(pat1[0])):
            corr += pat1[row_ind][col_ind] * pat2[row_ind][col_ind]
    return corr