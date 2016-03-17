import cPickle

f = file('ctr.save', 'rb')  # read binary
obj = cPickle.load(f)
f.close()

print "obj:", obj
