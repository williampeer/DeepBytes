from NeocorticalNetwork import *
from Tools import *

# def __init__(self, in_dim, h_dim, out_dim, alpha, momentum):
# Neocortical module:
ann = NeocorticalNetwork(49, 50, 49, 0.85, 0.01)

a = [0.1, 0.2] * 24
a.append(0.1)
I = np.asarray([a], dtype=np.float32)
b = [-0.2, -0.4] * 24
b.append(-.2)
O = np.asarray([b], dtype=np.float32)

iopair = [I, O]

show_image_from(I)
for i in range(20000):
    ann.train([iopair])
print ann.in_h_Ws.get_value()
print ann.h_out_Ws.get_value()
ann.print_layers()