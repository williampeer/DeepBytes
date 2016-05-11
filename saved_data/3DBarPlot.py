# Based on the source: https://pythonprogramming.net/3d-bar-charts-python-matplotlib/

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import Parser
import matplotlib.cm as cm

_x_width = 0.8


def bar_plot_3d_prr(prr_values):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    xpos, ypos, dz = unwrap_values(prr_values)
    num_of_values = len(xpos)
    zpos = np.zeros(num_of_values)
    dx = np.ones(num_of_values) * _x_width
    dy = np.ones(num_of_values) * _x_width

    cmaps = ['c', 'y', 'g', 'b']
    colours = []
    for x_val in xpos:
        colours.append(cmaps[int(x_val % 4)])
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colours, alpha=0.5)
    plt.xticks([2, 3, 4, 5], ('2x5', '3x5', '4x5', '5x5'))
    plt.xlabel('Set size')
    plt.ylabel('DG-weighting')
    plt.title('Recall ratio by set size and DG-weighting')
    plt.show()


def bar_plot_3d_prr_and_spurious(prr_values, spurious_values):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    xpos, ypos, dz = unwrap_values(prr_values)
    num_of_values = len(xpos)

    cmaps = ['c', 'y', 'g', 'b']
    colours = []
    for x_val in xpos:
        colours.append(cmaps[int(x_val % 4)])

    xpos_spur, ypos_spur, dz_spur = unwrap_values(spurious_values)
    for i in range(len(dz_spur)):
        dz_spur[i] = - dz_spur[i]
    num_of_values_spur = len(xpos_spur)
    zpos = np.zeros(num_of_values + num_of_values_spur)
    dx = np.ones(num_of_values + num_of_values_spur) * _x_width
    dy = np.ones(num_of_values + num_of_values_spur) * _x_width

    xpos = xpos + xpos_spur
    ypos = ypos + ypos_spur
    dz = dz + dz_spur
    print "lens:", len(xpos), len(ypos), len(zpos), len(dx), len(dy), len(dz)
    # colours = colours + num_of_values_spur * ['r']

    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', alpha=0.5)
    plt.xticks([2, 3, 4, 5], ('2x5', '3x5', '4x5', '5x5'))
    plt.yticks(np.arange(-1, 1.1, 0.1))
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
values_prrs, values_spurious = process_3d_data(parsed_data[:1200])
# bar_plot_3d_prr(prr_values=values_prrs)
bar_plot_3d_prr_and_spurious(prr_values=values_prrs, spurious_values=values_spurious)