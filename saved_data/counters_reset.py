import cPickle


def reset_counters():
    ctr = 0
    ctr_f = file('ctr.save', 'wb')
    cPickle.dump(ctr, ctr_f, protocol=cPickle.HIGHEST_PROTOCOL)
    ctr_f.close()

reset_counters()