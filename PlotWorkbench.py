import NeocorticalModuleTraining
import Tools
import NeocorticalMemoryConsolidation
from DataWrapper import training_patterns_associative as training_set

avgs_global = []
avgs_local = []
for set_size in range(2,6):
    original_training_set = training_set[:set_size * 5]

    global_gs = []
    local_gs = []

    for i in range(40):
        ann_global = NeocorticalModuleTraining.global_sequential_FFBP_training(ss=set_size, training_iterations=200)
        ann_local = NeocorticalModuleTraining.traditional_training_with_catastrophic_interference(
            ss=set_size, training_iterations=200)

        # global_io_results = Tools.generate_recall_attempt_results_for_ann(ann_global, original_training_set)
        # local_io_results = Tools.generate_recall_attempt_results_for_ann(ann_local, original_training_set)
        #
        # Tools.save_aggregate_image_from_ios(global_io_results, 'global_aggregate_im', 0)
        # Tools.save_aggregate_image_from_ios(local_io_results, 'local_aggregate_im', 1)

        global_goodness = NeocorticalMemoryConsolidation.evaluate_goodness_of_fit(ann_global, original_training_set)
        local_goodness = NeocorticalMemoryConsolidation.evaluate_goodness_of_fit(ann_local, original_training_set)
        global_gs.append(global_goodness)
        local_gs.append(local_goodness)

        log_line = 'EVALUATED baseline. g\'s - ' + 'global: ' + str(global_goodness) + ', local: ' + str(local_goodness)
        print log_line
        Tools.append_line_to_log(log_line)

    avg_global_g = Tools.get_avg(global_gs)
    avg_local_g = Tools.get_avg(local_gs)

    avgs_global.append(avg_global_g)
    avgs_local.append(avg_local_g)

    final_result_line = 'Final results for current set size: global avg. = ' + str(avg_global_g) + ', local avg. = ' + \
                        str(avg_local_g)
    print final_result_line
    Tools.append_line_to_log(final_result_line)
