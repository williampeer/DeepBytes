import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from PIL import Image
import numpy as np
import cPickle
import os

theano.config.floatX = 'float32'


def get_bipolar_in_out_values(values):
    new_values = np.ones_like(values, dtype=np.float32)
    for value_index in xrange(values.shape[1]):
        if values[0][value_index] < 0:
            new_values[0][value_index] = -1
    return new_values


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
    # print "Log: len(pattern_pairs):", len(pattern_pairs)
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    else:
        print "Error: Img-path already exists."

    img_ctr = 0
    for pattern in pattern_pairs:
        img_in = create_image_helper(pattern[0])
        img_out = create_image_helper(pattern[1])
        img_in.save(img_path+'/image#'+str(img_ctr)+'_input', 'PNG')
        img_out.save(img_path+'/image#'+str(img_ctr)+'_output', 'PNG')
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

random_f = theano.function([rows, columns], outputs=shared_random_generator.random_integers(
    size=(rows, columns), low=0, high=10000, dtype='float32')/10000.)


def get_random_input(in_dim):
    return 2 * binomial_f(1, in_dim, 0.5) - np.ones((1, in_dim), dtype=np.float32)


def set_contains_pattern(patterns_set, pattern):
    for pat in patterns_set:
        if get_pattern_correlation(pat, pattern) == 1:
            return True
    return False

pat1 = T.fmatrix()
pat2 = T.fmatrix()
get_pattern_correlation = theano.function([pat1, pat2], outputs=T.sum(pat1 * pat2)/(pat1.shape[0] * pat1.shape[1]))


def get_pattern_correlation_slow(pattern_1, pattern_2):
    corr = 0
    for row_ind in range(len(pattern_1)):
        for col_ind in range(len(pattern_1[0])):
            corr += pattern_1[row_ind][col_ind] * pattern_2[row_ind][col_ind]
    return corr


def save_experiment_4_1_results(hpc, rand_ins, all_chaotically_recalled_patterns, tar_patts, custom_name, train_set_size):
    experiment_dir = get_experiment_dir()

    unique_chaotically_recalled_patterns = []
    for pattern_set in all_chaotically_recalled_patterns:
        for p in pattern_set:
            if not set_contains_pattern(unique_chaotically_recalled_patterns, p):
                unique_chaotically_recalled_patterns.append(p)

    # write perfect recall rate to log:
    log_perfect_recall_rate(unique_chaotically_recalled_patterns, tar_patts)
    save_images_from(unique_chaotically_recalled_patterns, experiment_dir+'/images')

    # hpc_f = file(experiment_dir+'/hpc_'+custom_name+'.save', 'wb')
    # cPickle.dump(hpc, hpc_f, protocol=cPickle.HIGHEST_PROTOCOL)
    # hpc_f.close()

    # save HPC-info; write to file
    hpc_info_string = "epsilon: "+str(hpc._epsilon) + '\n' + \
                      "dg weighting: "+str(hpc._weighting_dg) + '\n' + \
                      "neuronal turnover ratio: "+str(hpc._turnover_rate) + '\n' + \
                      "number of distinct extracted patterns: " + str(len(unique_chaotically_recalled_patterns)) + \
                      "custom name: " + custom_name
    hpc_info_f = file(experiment_dir+'/hpc_info.txt', 'wb')
    hpc_info_f.write(hpc_info_string)
    hpc_info_f.close()

    f2 = file(get_chaotic_pat_dir(train_set_size)+'/_chaotically_recalled_patterns_exp#' +
              str(get_experiment_counter()) + '.save', 'wb')
    cPickle.dump(all_chaotically_recalled_patterns, f2, protocol=cPickle.HIGHEST_PROTOCOL)
    f2.close()

    f3 = file(get_chaotic_pat_dir(train_set_size)+'/_corresponding_random_ins_exp#' +
              str(get_experiment_counter()) + '.save', 'wb')
    cPickle.dump(rand_ins, f3, protocol=cPickle.HIGHEST_PROTOCOL)
    f3.close()


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

    # f = file(experiment_dir+'/information_vector'+custom_name+'.save', 'wb')
    # cPickle.dump(information_vector, f, protocol=cPickle.HIGHEST_PROTOCOL)
    # f.close()


def save_chaotic_recall_results(chaotically_recalled_patterns, pseudopatterns_I, pseudopatterns_II, original_patterns):
    experiment_dir = get_experiment_dir()

    unique_chaotically_recalled_outputs = []
    for pattern_set in chaotically_recalled_patterns:
        for pair in pattern_set:
            if not set_contains_pattern(unique_chaotically_recalled_outputs, pair[1]):
                unique_chaotically_recalled_outputs.append(pair[1])

    tar_patts = []
    for patt in original_patterns:
        tar_patts.append(patt[1])
    # write perfect recall rate to log:
    log_perfect_recall_rate(unique_chaotically_recalled_outputs, tar_patts)
    save_images_from(unique_chaotically_recalled_outputs, experiment_dir+'/images')

    f2 = file(get_chaotic_pat_dir(len(original_patterns)/5)+'/full_chaotically_recalled_patterns_exp#' +
              str(get_experiment_counter()) + '.save', 'wb')
    cPickle.dump(chaotically_recalled_patterns, f2, protocol=cPickle.HIGHEST_PROTOCOL)
    f2.close()

    f3 = file(get_chaotic_pat_dir(len(original_patterns)/5)+'/pseudopatterns_exp#' +
              str(get_experiment_counter()) + '.save', 'wb')
    cPickle.dump([pseudopatterns_I, pseudopatterns_II], f3, protocol=cPickle.HIGHEST_PROTOCOL)
    f3.close()

    p_I_unwrapped = []
    p_II_unwrapped = []
    for p_I_set in pseudopatterns_I:
        p_I_unwrapped += p_I_set
    for p_II_set in pseudopatterns_II:
        p_II_unwrapped += p_II_set

    save_images_from_pairs(p_I_unwrapped, experiment_dir+'/pseudopatterns_I')
    save_images_from_pairs(p_II_unwrapped, experiment_dir+'/pseudopatterns_II')


def get_experiment_counter():
    experiment_ctr_f = file('saved_data/ctr.save', 'rb')
    experiment_ctr = cPickle.load(experiment_ctr_f)
    experiment_ctr_f.close()

    return experiment_ctr


def get_parameter_counter():
    param_ctr_f = file('saved_data/parameter_ctr.save', 'rb')
    param_ctr = cPickle.load(param_ctr_f)
    param_ctr_f.close()

    increment_param_counter()

    return param_ctr


def increment_experiment_counter():
    experiment_ctr_f = file('saved_data/ctr.save', 'rb')
    experiment_ctr = cPickle.load(experiment_ctr_f)
    experiment_ctr_f.close()

    experiment_ctr += 1

    experiment_ctr_f = file('saved_data/ctr.save', 'wb')
    cPickle.dump(experiment_ctr, experiment_ctr_f, protocol=cPickle.HIGHEST_PROTOCOL)
    experiment_ctr_f.close()


def increment_param_counter():
    experiment_ctr_f = file('saved_data/parameter_ctr.save', 'rb')
    experiment_ctr = cPickle.load(experiment_ctr_f)
    experiment_ctr_f.close()

    experiment_ctr += 1

    experiment_ctr_f = file('saved_data/parameter_ctr.save', 'wb')
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


def get_chaotic_pat_dir(set_size):
    experiment_dir = 'saved_data/chaotic_pattern_recalls_set_size_'+str(set_size)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    return experiment_dir


def append_line_to_log(line):
    log_path = 'saved_data/log.txt'
    file_contents = ""

    if os.path.exists(log_path):
        log_f = file(log_path, 'rb')
        file_contents = log_f.read()
        log_f.close()

    file_contents += line + '\n'

    log_f = file(log_path, 'wb')
    log_f.write(file_contents)
    log_f.close()


def log_perfect_recall_rate(hipp_chaotic_outputs, train_set):
    log_path = 'saved_data/log.txt'
    file_contents = ""

    if os.path.exists(log_path):
        log_f = file(log_path, 'rb')
        file_contents = log_f.read()
        log_f.close()

    perf_recalls = 0
    for p in hipp_chaotic_outputs:
        if set_contains_pattern(train_set, p):
            perf_recalls += 1

    perfect_recall_rate = perf_recalls / float(len(train_set))

    file_contents += "Perfect recall rate: " + "{:6.3f}".format(perfect_recall_rate) + '\n'
    file_contents += "Spurious patterns: " + str(len(hipp_chaotic_outputs) - perf_recalls) + '\n'

    log_f = file(log_path, 'wb')
    log_f.write(file_contents)
    log_f.close()


def flip_bits_f(input_vector, flip_P):
    flip_bits = np.ones_like(input_vector[0], dtype=np.float32) - 2 * binomial_f(1, len(input_vector[0]), flip_P)
    return input_vector * flip_bits  # binomial_f returns a 2-dim. array


def set_to_equal_parameters(hpc, test_hpc):
    test_hpc.in_ec_weights = hpc.in_ec_weights
    test_hpc.ec_dg_weights = hpc.ec_dg_weights
    test_hpc.ec_ca3_weights = hpc.ec_ca3_weights
    test_hpc.dg_ca3_weights = hpc.dg_ca3_weights
    test_hpc.ca3_ca3_weights = hpc.ca3_ca3_weights
    test_hpc.ca3_out_weights = hpc.ca3_out_weights

    return test_hpc


def network_visualization(hpc):
    # weights_image = Image.new('1', ())
    pass


def generate_recall_attempt_results(hpc, training_patterns):
    io_trials = []
    for pattern in training_patterns:
        current_letter_ios = []
        for i in range(15):  # recall for 15 iterations, save IO every time
            current_letter_ios.append([pattern[0], hpc.recall_for_i_iters_with_input(pattern[0], num_of_iterations=1)])
        io_trials.append(current_letter_ios)
    # generate aggregate image of all outputs.
    return io_trials


def save_aggregate_image_from_ios(ios, im_name, im_ctr):
    num_of_ims = len(ios)
    aggregate_im = Image.new('1', (7 * 8 * num_of_ims + num_of_ims+2, 7 * 8 + 2))
    for IO_ctr in range(num_of_ims):
        out = ios[IO_ctr][1]
        current_im = create_image_helper(out)
        aggregate_im.paste(current_im, (1 + 7 * 8 * IO_ctr + IO_ctr, 1))
    # aggregate_im.show()
    # save to disk
    if not os.path.exists('saved_data/aggregate_output_images'):
        os.mkdir('saved_data/aggregate_output_images')
    aggregate_im.save('saved_data/aggregate_output_images/'+im_name+'#'+str(im_ctr)+'.png', 'PNG')


def retrieve_patterns_for_consolidation(exp_num, set_size):
    file_path = 'saved_data/current-consolidation-path/' + 'chaotic_pattern_recalls_set_size' + str(set_size)
    chaotic_patterns_filename = '/full_chaotically_recalled_patterns_exp#' + str(exp_num)
    pseudopatterns_filename = '/pseudopatterns_exp#' + str(exp_num)

    chaotic_file = file(file_path+chaotic_patterns_filename, 'rb')
    pseudopatterns_file = file(file_path+pseudopatterns_filename, 'rb')

    chaotic_patterns = cPickle.load(chaotic_file)
    pseudpatterns = cPickle.load(pseudopatterns_file)

    return [chaotic_patterns, pseudpatterns]
