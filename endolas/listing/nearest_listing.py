import math
import copy

# ------------------------------------------------------------------------------
def nearest_neighbor_kernel(warped_key_2_warped_val, fixed_key_2_fixed_val,
                            scale_factor_x, scale_factor_y):
    """ Based on x- and y-coordinates of two different sets of points find a
        mapping between both that is based on the nearest neighbor.

    :param dict warped_key_2_warped_val: Warped keypoints and their
                                         x,y-coordinates stored as '[x,y]'.
    :param dict fixed_key_2_fixed_val: Fixed keypoints and their
                                       x,y-coordinates stored as '[x,y]'.
    :param float scale_factor_x: A scaling factor for the x-coordinate that
                                 is the width of the warped image
                                 space over the width of the fixed image space.
    :param float scale_factor_y: A scaling factor for the y-coordinate that
                                 is the height of the warped image
                                 space over the height of the fixed image space.
    :return: A mapping from warped_keys, that are identifiers for
             detected keypoints, to fixed_keys, that are the identifiers in the
              regular spaced grid and can be interpreted as classes.
    :rtype: dict
    """
    # 0) Compute nearest neighbor
    warped_key_2_fixed_key = dict()
    warped_key_2_warped_val = copy.deepcopy(warped_key_2_warped_val)
    fixed_key_2_fixed_val = copy.deepcopy(fixed_key_2_fixed_val)
    is_search_finished = False

    while not is_search_finished:
        key_warped_2_nearest_neighbor = dict()
        key_warped_2_nearest_distance = dict()
        nearest_fixed_neighbor_2_key_warpeds = dict()

        for key_warped, value_warped in warped_key_2_warped_val.items():
            nearest_fixed_neighbor = None
            nearest_distance = math.inf

            for key_fixed, value_fixed in fixed_key_2_fixed_val.items():
                val_fix_0 = value_fixed[0] * scale_factor_x
                val_fix_1 = value_fixed[1] * scale_factor_y

                distance = math.sqrt(
                    (value_warped[0] - val_fix_0) ** 2 + (value_warped[1] -
                                                          val_fix_1) ** 2)

                if distance < nearest_distance:
                    nearest_fixed_neighbor = key_fixed
                    nearest_distance = distance

            key_warped_2_nearest_neighbor[key_warped] = nearest_fixed_neighbor
            key_warped_2_nearest_distance[key_warped] = nearest_distance

            try:
                nearest_fixed_neighbor_2_key_warpeds[nearest_fixed_neighbor]\
                    .append(key_warped)

            except KeyError:
                nearest_fixed_neighbor_2_key_warpeds[nearest_fixed_neighbor] = \
                    [key_warped]

        # 1) Evaluate all found neighbors
        for nearest_fixed_neighbor, key_warpeds in \
                nearest_fixed_neighbor_2_key_warpeds.items():
            nearest_warped_neighbor = None
            nearest_distance = math.inf

            for key_warped in key_warpeds:
                if key_warped_2_nearest_distance[key_warped] < nearest_distance:
                    nearest_distance = key_warped_2_nearest_distance[key_warped]
                    nearest_warped_neighbor = key_warped

            if nearest_warped_neighbor != None:
                _ = warped_key_2_warped_val.pop(nearest_warped_neighbor)
                _ = fixed_key_2_fixed_val.pop(nearest_fixed_neighbor)
                warped_key_2_fixed_key[nearest_warped_neighbor] = \
                    nearest_fixed_neighbor

        # 2) Determine loop criterion
        if len(warped_key_2_warped_val) == 0 or len(fixed_key_2_fixed_val) == 0:
            is_search_finished = True

    return warped_key_2_fixed_key
