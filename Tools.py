import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from PIL import Image
import numpy as np
import cPickle
import os

theano.config.floatX = 'float32'


def show_image_from(out_now):
    im = create_image_helper(out_now)
    im.show()
    # print "Output image"


def save_images_from(single_patterns, img_path):
    img_ctr = 0

    if not os.path.exists(img_path):
        os.mkdir(img_path)
    else:
        print "Error: Img-path already exists."

    for pattern in single_patterns:
        img = create_image_helper(pattern)
        img.save(img_path+'/image#'+str(img_ctr), 'BMP')
        img_ctr += 1


def save_images_from_pairs(pattern_pairs, img_path):
    img_ctr = 0

    if not os.path.exists(img_path):
        os.mkdir(img_path)
    else:
        print "Error: Img-path already exists."

    for pattern in pattern_pairs:
        img_in = create_image_helper(pattern[0])
        img_out = create_image_helper(pattern[1])
        img_in.save(img_path+'/image#'+str(img_ctr)+'_input', 'BMP')
        img_out.save(img_path+'/image#'+str(img_ctr)+'_output', 'BMP')
        img_ctr += 1


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


def show_image_ca3(in_values):
    pattern = np.asarray(in_values, dtype=np.float32)
    width = 24
    height = 20
    pixel_scaling_factor = 2 ** 3
    im = Image.new('1', (width*pixel_scaling_factor, height*pixel_scaling_factor))
    for element in range(pattern.shape[1]):
        for i in range(pixel_scaling_factor):
            for j in range(pixel_scaling_factor):
                im.putpixel(((element % width)*pixel_scaling_factor + j,
                             np.floor(element/width).astype(np.int8) * pixel_scaling_factor + i),
                            pattern[0][element] * 255)
    im.show()


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

random_f = theano.function([rows, columns], outputs=shared_random_generator.
                           random_integers(size=(rows, columns), low=-100000, high=100000, dtype='float32')/100000.)


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
    experiment_dir = get_experiment_dir()

    hpc_f = file(experiment_dir+'/hpc_'+custom_name+'.save', 'wb')
    cPickle.dump(hpc, hpc_f, protocol=cPickle.HIGHEST_PROTOCOL)
    hpc_f.close()

    save_images_from(chaotically_recalled_patterns, experiment_dir+'/images')

    f2 = file(experiment_dir+'/_chaotically_recalled_patterns.save', 'wb')
    cPickle.dump(chaotically_recalled_patterns, f2, protocol=cPickle.HIGHEST_PROTOCOL)
    f2.close()


def save_experiment_4_2_results(information_vector, custom_name):
    experiment_dir = get_experiment_dir()

    pseudopatterns_I = information_vector[0]
    pseudopatterns_II = information_vector[1]
    save_images_from_pairs(pseudopatterns_I, experiment_dir+'/pseudopatterns_I')
    save_images_from_pairs(pseudopatterns_II, experiment_dir+'/pseudopatterns_II')

    neocortically_recalled_IOs = information_vector[2]
    save_images_from_pairs(neocortically_recalled_IOs, experiment_dir+'/neocortical_recall')

    f_goodness = file(experiment_dir+'/goodness_of_fit.txt', 'w')
    f_goodness.write(str(information_vector[4]) + ' <-- goodness of fit')  # goodness of fit
    f_goodness.close()

    f = file(experiment_dir+'/information_vector'+custom_name+'.save', 'wb')
    cPickle.dump(information_vector, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def get_experiment_counter():
    experiment_ctr_f = file('saved_data/ctr.save', 'rb')
    experiment_ctr = cPickle.load(experiment_ctr_f)
    experiment_ctr_f.close()

    return experiment_ctr


def increment_experiment_counter():
    experiment_ctr_f = file('saved_data/ctr.save', 'rb')
    experiment_ctr = cPickle.load(experiment_ctr_f)
    experiment_ctr_f.close()

    experiment_ctr += 1

    experiment_ctr_f = file('saved_data/ctr.save', 'wb')
    cPickle.dump(experiment_ctr, experiment_ctr_f, protocol=cPickle.HIGHEST_PROTOCOL)
    experiment_ctr_f.close()


def get_experiment_dir():
    experiment_counter = get_experiment_counter()
    experiment_dir = 'saved_data/experiment#'+str(experiment_counter)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    else:
        print "Info.: OS path already exists."

    return experiment_dir