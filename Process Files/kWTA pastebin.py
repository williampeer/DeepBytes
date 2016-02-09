# def kWTA(self, values, f_r):
#     # print "values[0]", values[0]
#     values_length = len(values[0])
#     k = np.round(values_length * f_r).astype(np.int32)
#
#     sort_values_f = theano.function([], outputs=T.sort(values))
#     sorted_values = sort_values_f()
#     k_th_largest_value = sorted_values[0][values_length-k-1]
#
#     threshold = 1.0
#     if k_th_largest_value == -1.0 and sorted_values[0][values_length-1] == -1.0:
#             return binomial_f(1, values_length, f_r)
#
#     if k_th_largest_value == 1.0:  # or k_th_largest_value == -1:  # as this occurs, it seems that weights converge
#         if sorted_values[0][0] == 1.0:
#             return binomial_f(1, values_length, f_r)
#
#         ctr = 0
#         cur_value = sorted_values[0][values_length-k-1-ctr]
#         while cur_value == 1:
#             cur_value = sorted_values[0][values_length-k-1-ctr]
#             ctr += 1
#
#         threshold = np.divide(k.astype(np.float32), (ctr+k))
#         print threshold
#
#     # towards weights/elements all -1, or all 1. TODO: Fix this.
#     # print "sorted_values:", sorted_values
#     print "k_th_largest_value:", k_th_largest_value
#
#     mask_vector = k_th_largest_value * np.ones_like(values)
#     result = (values >= mask_vector).astype(np.float32)
#
#     if threshold < 1.0:
#         for i in range(values_length):
#             if result[0][i] == 0:
#                 if np.random.random() < (1-threshold):
#                     result[0][i] = 0
#     # print result
#
#     return result


# def kWTA(self, values, firing_rate):
#         print "values:", values
#         # tuples = []
#         # index_ctr = 0
#         # for value in values[0]:
#         #     tuples.append((value, index_ctr))
#         #     index_ctr += 1
#
#         # kWTA EC:
#         k_neurons = np.floor(len(values[0]) * firing_rate).astype(np.int32)  # k determined by the firing rate
#         # k_neurons = int(len(values[0]) * firing_rate)  # k determined by the firing rate
#
#         sort_act_vals = theano.function([], outputs=T.sort(tuples))
#         act_vals_sorted = sort_act_vals()
#         k_th_largest_act_val = act_vals_sorted[len(values[0])-1 - k_neurons]  # TODO: Check that it is sorted in an ascending order.
#         print "act_vals_sorted:", act_vals_sorted
#         print "k_th_largest_act_val:", k_th_largest_act_val
#
#
#
#         # # TODO: Build hash-map. Draw random for same value 'til k nodes drawn.
#         # # TODO: Check if source for this bug stems from weights. Perhaps execute equations on paper?
#         #
#         # new_values = np.zeros_like(values, dtype=np.float32)
#         #
#         # for act_val_index in range(len(values[0])):
#         #     if values[0][act_val_index] > k_th_largest_act_val:
#         #         new_values[0][act_val_index] = 1
#         #     elif values[0][act_val_index] == k_th_largest_act_val:
#         #         if np.sum(new_values[0]) < k_neurons:
#         #             new_values[0][act_val_index] = 1
#         #         else:
#         #             return new_values
#         #
#         # return new_values


# def kWTA(self, values, f_r):
#         # print "values[0]", values[0]
#         values_length = len(values[0])
#         k = np.round(values_length * f_r).astype(np.int32)
#         values_sum = np.sum(values[0])
#         print "values_sum:", values_sum, "k:", k
#         # print "values_length:", values_length
#         # edge cases. note that the sum may be 0 or the length sometimes too without the edge case.
#         if values_sum == values_length or values_sum == 0:
#             print "equal sum to length or 0"
#             all_zero_or_one = True
#             for el in values[0]:
#                 if el != 0 and el != 1:
#                     # print "this el voiasdoipasd:", el
#                     all_zero_or_one = False
#                     print "all zero or one false"
#                     break
#             if all_zero_or_one:  # return random indices as on (1)
#                 return binomial_f(1, values_length, f_r)
#
#         sort_values = theano.function([], outputs=T.sort(values))
#         sorted_values = sort_values()
#         k_th_largest_value = sorted_values[0, values_length-k-1]
#
#         new_values = np.zeros_like(values)
#         k_ctr = 0
#         ind_ctr = 0
#         for el in values[0]:
#             if el > k_th_largest_value:
#                 new_values[0][ind_ctr] = 1
#                 k_ctr += 1
#             elif el == k_th_largest_value:
#                 if k_ctr < k:
#                     new_values[0][ind_ctr] = 1
#                     k_ctr += 1
#                 else:
#                     break
#             ind_ctr += 1
#
#         print "new_values:", new_values
#         print "np.sum(new_values):", np.sum(new_values)
#         return new_values