import copy

# ------------------------------------------------------------------------------
def bubblesort(list1, list2):
    """ Takes two list and sorts both in ascending order according to the values
        in list1 with a bubblesort algorithm.

    :param list list1: A list on whose content the sorting is based.
    :param list list2: A list that is sorted based on list1.
    :return: A tuple containing the sorted list1 and list2
    :rtype: tuple
    """
    list_length = len(list1)
    list1_sorted = copy.deepcopy(list1)
    list2_sorted = copy.deepcopy(list2)

    # Traverse all list elements, range(list_length) also okay, but one more
    # iteration.
    for i in range(list_length - 1):
        swap_counter = 0
        for j in range(0, list_length - i - 1):
            # Check if greater and swap entries in both lists
            if list1_sorted[j] > list1_sorted[j + 1]:
                list1_sorted[j], list1_sorted[j + 1] = list1_sorted[j + 1], \
                                                       list1_sorted[j]
                list2_sorted[j], list2_sorted[j + 1] = list2_sorted[j + 1], \
                                                       list2_sorted[j]
                swap_counter += 1

        if swap_counter == 0:
            break

    return list1_sorted, list2_sorted
