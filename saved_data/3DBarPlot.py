# Based on the source: https://pythonprogramming.net/3d-bar-charts-python-matplotlib/

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import Parser

_x_width = 0.8


def bar_plot_3d(values_3d):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    xpos, ypos, dz = unwrap_values(values_3d)
    num_of_values = len(xpos)
    zpos = np.zeros(num_of_values)
    dx = np.ones(num_of_values) * _x_width
    dy = np.ones(num_of_values) * _x_width

    cmaps = ['r', 'b', 'c', 'y']
    colours = []
    for x_val in xpos:
        colours.append(cmaps[int(x_val % 4)])
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colours, alpha=0.5)
    plt.xticks([2, 3, 4, 5], ('2x5', '3x5', '4x5', '5x5'))
    plt.xlabel('Set size')
    plt.ylabel('DG-weighting')
    plt.title('Recall ratio by set size and DG-weighting')
    plt.show()


def process_3d_data(current_data):
    prr_by_dgw_list, spurious_by_dgw_list, stds_prr, stds_spurious = \
        Parser.get_avg_perfect_recall_and_avg_spurious_recall_from_data_for_dg_ws(current_data, iterations_per_dg_w=10,
                                                                                  dg_ws=30)

    print "spurious_by_dgw_list:", spurious_by_dgw_list
    values_prrs = []
    values_spurious = []

    dgw_ctr = 0
    for avg_list in prr_by_dgw_list:
        x_ctr = 2
        for avg_val in avg_list:
            x = x_ctr - _x_width/2.
            y = dgw_ctr
            z = avg_val
            values_prrs.append([x, y, z])

            x_ctr += 1

        dgw_ctr += 1

    dgw_ctr = 0
    for avg_list in spurious_by_dgw_list:
        x_ctr = 2
        for avg_val in avg_list:
            x = x_ctr - _x_width/2.
            y = dgw_ctr
            z = avg_val
            values_spurious.append([x, y, z])

            x_ctr += 1

        dgw_ctr += 1

    return [values_prrs, values_spurious]


def unwrap_values(values_3d):
    xpos = []
    ypos = []
    zpos = []

    for tuple in values_3d:
        xpos.append(tuple[0])
        ypos.append(tuple[1])
        zpos.append(tuple[2])

    return [xpos, ypos, zpos]

log_filename = 'dgw-exps-corrected.txt'
parsed_data = Parser.get_data_from_log_file(log_filename)
values_prrs, values_spurious = process_3d_data(parsed_data[2400:3600])
bar_plot_3d(values_prrs)
bar_plot_3d(values_spurious)
