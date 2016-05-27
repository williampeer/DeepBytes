# Based on the source: https://pythonprogramming.net/3d-bar-charts-python-matplotlib/

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import Parser

_x_width = 0.8


def bar_plot_3d(values_3d, title, x_ticks_labels, y_label, y_ticks_values, y_ticks_labels, opacity_value):
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
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colours, alpha=opacity_value)
    ax1.set_zlabel('Recall ratio')
    plt.xticks([2, 3, 4, 5], x_ticks_labels)
    plt.xlabel('Set size')
    plt.yticks(y_ticks_values, y_ticks_labels)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def process_3d_data(current_data, iterations_per_config, num_of_configs):
    prr_by_config_list, spurious_by_config_list, stds_prr, stds_spurious = \
        Parser.get_avg_perfect_recall_and_avg_spurious_recall_from_data_for_configs(
            current_data, iterations_per_config=iterations_per_config, num_of_configs=num_of_configs)

    # print "spurious_by_dgw_list:", spurious_by_dgw_list
    values_prrs = []
    values_spurious = []

    config_ctr = 0
    for avg_list in prr_by_config_list:
        x_ctr = 2
        for avg_val in avg_list:
            x = x_ctr - _x_width/2.
            y = config_ctr
            z = avg_val
            values_prrs.append([x, y, z])

            x_ctr += 1

        config_ctr += 1

    config_ctr = 0
    for avg_list in spurious_by_config_list:
        x_ctr = 2
        for avg_val in avg_list:
            x = x_ctr - _x_width/2.
            y = config_ctr
            z = avg_val
            values_spurious.append([x, y, z])

            x_ctr += 1

        config_ctr += 1

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

log_filename = 'Logs/relaxed-criterion-logs/homo/global-sync-async-50iters.txt'
log_filename2 = 'Logs/relaxed-criterion-logs/homo/local-sync-50iters.txt'
parsed_data = Parser.get_data_from_log_file_i_iters_schemes(log_filename, 50)  # 1 config.
parsed_data2 = Parser.get_data_from_log_file_i_iters_schemes(log_filename2, 5)  # 2 configs.
aggregate_parsed_data = parsed_data2 + parsed_data

values_prrs, values_spurious = process_3d_data(aggregate_parsed_data, iterations_per_config=20, num_of_configs=3)
# values_prrs, values_spurious = process_3d_data(aggregate_parsed_data, iterations_per_config=20, num_of_configs=4)
title = 'Recall ratio by set size and model configuration (50 iterations)'
title_spurious = 'Non-perfect recall ratio by set size and configuration (50 iterations)'
y_label = 'Configurations'
num_of_configs = 5
y_ticks_values = np.arange(num_of_configs)+np.ones(num_of_configs)*0.5
y_ticks_labels = ('(local) Sync., '+r'$\tau$=0.50'+',\ntm=1, DGW=25',
                  '(global) Sync., '+r'$\tau$=0.50'+',\ntm=1, DGW=25',
                  '(global) Async.,'+r'$\tau=0.04$,'+'\ntm=0, DGW=25')
x_ticks_labels = ('2x5', '3x5', '4x5', '5x5')  # local
# # x_ticks_labels = ('10', '15', '20', '25')  # global
bar_plot_3d(values_prrs, title, x_ticks_labels, y_label, y_ticks_values, y_ticks_labels, opacity_value=0.5)
bar_plot_3d(values_spurious, title_spurious, x_ticks_labels, y_label, y_ticks_values, y_ticks_labels, opacity_value=0.5)

# values_prrs, values_spurious = process_3d_data(parsed_data[3600:4800], iterations_per_config=10, num_of_configs=30)
# title = 'Recall ratio by set size and DG-weighting'
# title_spurious = 'Non-perfect recall ratio by set size and configuration'
# y_label = 'DG-weighting'
# y_ticks_values = np.arange(30)+np.ones(30)*0.5
# y_ticks_labels = range(30)
# bar_plot_3d(values_prrs, title, x_ticks_labels, y_label, y_ticks_values, y_ticks_labels, opacity_value=0.2)
# bar_plot_3d(values_spurious, title_spurious, x_ticks_labels, y_label, y_ticks_values, y_ticks_labels, opacity_value=0.2)
