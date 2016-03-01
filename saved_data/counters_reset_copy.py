import cPickle


def reset_counters():
    ctr = 0
    ctr_f = file('ctr.save', 'wb')
    cPickle.dump(ctr, ctr_f, protocol=cPickle.HIGHEST_PROTOCOL)
    ctr_f.close()

    info_ctr = 0
    info_f = file('information-ctr.save', 'wb')
    cPickle.dump(info_ctr, info_f, protocol=cPickle.HIGHEST_PROTOCOL)
    info_f.close()

    img_ctr = 0
    img_f = file('image-ctr.save', 'wb')
    cPickle.dump(img_ctr, img_f, protocol=cPickle.HIGHEST_PROTOCOL)
    img_f.close()

reset_counters()