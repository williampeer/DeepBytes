import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from PIL import Image
import numpy as np
import cPickle

theano.config.floatX = 'float32'


def show_image_from(out_now):
    im = create_image_helper(out_now)
    im.show()
    # print "Output image"


def save_images_from(patterns):
    ctr_f = file('saved_data/ctr.save', 'rb')
    ctr_exp_1 = cPickle.load(ctr_f)
    ctr_f.close()

    info_f = file('saved_data/information-ctr.save', 'rb')
    ctr_exp_2 = cPickle.load(info_f)
    info_f.close()

    experiment_ctr = str(ctr_exp_1) + str(ctr_exp_2)

    img_f = file('saved_data/image-ctr.save', 'rb')
    img_ctr = cPickle.load(img_f)
    img_f.close()

    for pattern in patterns:
        img = create_image_helper(pattern)
        img.save('saved_data/images/experiment#'+experiment_ctr+'image#'+str(img_ctr), 'BMP')
        img_ctr += 1

    img_f = file('saved_data/image-ctr.save', 'wb')
    cPickle.dump(img_ctr, img_f, protocol=cPickle.HIGHEST_PROTOCOL)
    img_f.close()


def create_image_helper(in_values):
    pattern = np.asarray(in_values, dtype=np.float32)
    width = 7
    height = 7
    pixel_scaling_factor = 2 ** 3  # Exponent of two for symmetry.
    im = Image.new('1', (width*pixel_scaling_factor, height*pixel_scaling_factor))
    for element in range(pattern.shape[1]):
        for i in range(pixel_scaling_factor):
            for j in range(pixel_scaling_factor):
                im.putpixel(((element % width)*pixel_scaling_factor + j,
                             np.floor(element/height).astype(np.int8) * pixel_scaling_factor + i),
                            pattern[0][element] * 255)
    return im


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


def save_experiment_4_1_results(hpc, chaotically_recalled_patterns, custom_name):
    ctr_f = file('saved_data/ctr.save', 'rb')
    storage_counter = cPickle.load(ctr_f)
    ctr_f.close()

    storage_counter += 1
    f = file('saved_data/'+custom_name+'hpc-serialized#'+str(storage_counter)+'.save', 'wb')
    cPickle.dump(hpc.serialize_hpc(), f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    f2 = file('saved_data/'+custom_name+'chaotically_recalled_patterns#'+str(storage_counter)+'.save', 'wb')
    cPickle.dump(chaotically_recalled_patterns, f2, protocol=cPickle.HIGHEST_PROTOCOL)
    f2.close()

    ctr_f = file('saved_data/ctr.save', 'wb')
    cPickle.dump(storage_counter, ctr_f, protocol=cPickle.HIGHEST_PROTOCOL)
    ctr_f.close()


def save_experiment_4_2_results(information_vector, custom_name):
    ctr_f = file('saved_data/information-ctr.save', 'rb')
    info_counter = cPickle.load(ctr_f)
    ctr_f.close()

    info_counter += 1
    f = file('saved_data/'+custom_name+'information_vector.save', 'wb')
    cPickle.dump(information_vector, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    ctr_f = file('saved_data/information-ctr.save', 'wb')
    cPickle.dump(info_counter, ctr_f, protocol=cPickle.HIGHEST_PROTOCOL)
    ctr_f.close()
