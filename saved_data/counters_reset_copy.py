import cPickle


def reset_counters():
    ctr = 56
    ctr_f = file('ctr.save', 'wb')
    cPickle.dump(ctr, ctr_f, protocol=cPickle.HIGHEST_PROTOCOL)
    ctr_f.close()

    param_ctr = 56
    param_ctr_f = file('parameter_ctr.save', 'wb')
    cPickle.dump(param_ctr, param_ctr_f, protocol=cPickle.HIGHEST_PROTOCOL)
    param_ctr_f.close()

    # img_ctr = 0
    # img_f = file('image-ctr.save', 'wb')
    # cPickle.dump(img_ctr, img_f, protocol=cPickle.HIGHEST_PROTOCOL)
    # img_f.close()

reset_counters()
