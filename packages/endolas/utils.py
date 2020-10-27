from tensorflow import keras
from glob import glob
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import albumentations as albu
#import seaborn as sns
import tensorflow

import os
import json
import csv
import math
import copy
import imageio
import h5py

from pdb import set_trace
from .styles import *

# ----------------------------------------------------------------------------------------------------------------------
def _init_plot():
    """ Initialize all plot settings @ default dpi=100.
    """
    #sns.set_style('white')
    #sns.set_style('ticks')

    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['axes.labelsize'] = 8
    #plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['figure.figsize'] = 3, 1.5 # 3,2
    plt.rcParams['lines.linewidth'] = 1.
    # INFO: Default dpi = 100


# ----------------------------------------------------------------------------------------------------------------------
def show(image):
    """ Display only an image.

    :param PIL image:
        Single image to be displayed
    """
    plt.figure()
    plt.axis('off')
    plt.imshow(image, cmap='gist_yarg')
    plt.show()


def plot_convergence(paths, series, epochs=300, sigma=3, append='', plot1=-1, plot2=0, ylabel='MED', log=False,
                     lower_limit=0.0, upper_limit=100.0):
    """ Create a convergence plot.

    :param dictionary paths: All experiment ids mapped to corresponding path including data
    :param int series: The series which was selected
    :param int epochs: Number of epochs to plot
    :param float sigma: Parameter for convolution filter
    :param bool normalize: Whether to normalize the data or not
    :param bool test: Whether data is based on test or validation set
    :param int plot: The index of the row to be plotted
    :param str label: The label to be displayed for y axis
    :param bool log: Whether to plot as semilog or not
    :param int upper_limit: The upper limit to plot on the y-axis
    """
    _init_plot()

    #if not normalize:
        #plt.axhline(1, color='k', linewidth=0.5)

    for experiment_id in paths.keys():
        afile = open(paths[experiment_id])
        areader = csv.reader(afile, delimiter=',')

        y1 = np.zeros(epochs)
        y2 = np.zeros(epochs)
        x = np.zeros(epochs)

        label1 = EX_2_ID_2_LABEL_1[series][experiment_id]
        color1 = EX_2_ID_2_COLOR_1[series][experiment_id]
        style1 = EX_2_ID_2_STYLE_1[series][experiment_id]

        label2 = EX_2_ID_2_LABEL_2[series][experiment_id]
        color2 = EX_2_ID_2_COLOR_2[series][experiment_id]
        style2 = EX_2_ID_2_STYLE_2[series][experiment_id]

        for index, row in enumerate(areader):
            if index == 0:
                continue

            if index == epochs + 1:
                break

            y1[index - 1] = row[plot1]
            y2[index - 1] = row[plot2]
            x[index - 1] = row[0]

        if sigma > 0.0:
            y1 = gaussian_filter1d(y1, sigma, mode='nearest')
            if plot2:
                y2 = gaussian_filter1d(y2, sigma, mode='nearest')

        if log:
            plt.semilogy(x, y1, label=label1, color=color1, linestyle=style1)
            if plot2:
                plt.semilogy(x, y2, label=label2, color=color2, linestyle=style2)

        else:
            plt.plot(x, y1, label=label1, color=color1, linestyle=style1)
            if plot2:
                plt.plot(x, y2, label=label2, color=color2, linestyle=style2)

    plt.ylabel('{}'.format(ylabel))

    afile.close()

    plt.xlabel('Epoch')
    #plt.legend(loc='upper right')

    plt.gca().spines['left'].set_position(('outward', 5))
    plt.gca().spines['bottom'].set_position(('outward', 5))

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # handles, labels = plt.gca().get_legend_handles_labels()
    # new_handles = []
    # new_handles.append(handles[0])
    # new_handles.append(handles[2])
    # new_handles.append(handles[4])
    # new_handles.append(handles[1])
    # new_handles.append(handles[3])
    # new_handles.append(handles[5])
    # new_labels = []
    # new_labels.append(labels[0])
    # new_labels.append(labels[2])
    # new_labels.append(labels[4])
    # new_labels.append(labels[1])
    # new_labels.append(labels[3])
    # new_labels.append(labels[5])
    # plt.legend(new_handles, new_labels, loc=(-0.1, 1.05), ncol=2, frameon=False, borderpad=0.0, labelspacing=0.2, handlelength=1.0, columnspacing=0.5)



    plt.legend(loc=(0.0, 1.05), ncol=2, frameon=False, borderpad=0.0, labelspacing=0.2, handlelength=1.0, columnspacing=0.5)

    plt.xlim([0, 100])
    plt.xticks([0, 20, 40, 60, 80, 100])
    plt.ylim([lower_limit, upper_limit])

    save_title = 'convergence'

    save_title += '_' + str(series)
    save_title += '_' + ylabel

    for experiment_id in paths.keys():
        save_title += '_' + str(experiment_id)

    save_title += append

    #plt.savefig(save_title + '.png', format='png', bbox_inches='tight', dpi=200)
    plt.savefig(save_title + '.svg', format='svg', bbox_inches='tight')


def get_augmenter(rotation=True, flip=True, keypoints=False):
    """ Get an augmenter.

    :param bool rotation: Property whether to include rotation or not
    :param bool rotation: Property whether to include flipping or not
    :param bool keypoints: Property whether to augment keypoints or not
    :return: A Compose object of the albumentations package
    :rtype: object
    """
    transformers = [albu.RandomBrightnessContrast(p=0.75),
                    albu.RandomGamma(p=0.75),
                    albu.Blur(p=0.5),
                    albu.GaussNoise(p=0.5)]

    if flip:
        transformers.append(albu.HorizontalFlip(p=0.5))

    if rotation:
        transformers.append(albu.Rotate(limit=30, border_mode=0, p=0.75))

    if keypoints:
        augmenter = albu.Compose(transformers, keypoint_params=albu.KeypointParams(format='xy'), p=1)
    else:
        augmenter = albu.Compose(transformers, p=1)

    return augmenter


def apply_smoothing(image, sigma=1.0, sigma_back=10.0):
    """ Smooth an image with two gaussian kernels.

    :param ndarray image: The image to be smoothed
    :param float sigma: The standard deviation for the kernel
    :param float sigma_back: The standard deviation for the kernel in the background
    :return: The smoothed image
    :rtype: ndarray
    """
    image_orig = image
    image = gaussian_filter(image, sigma=sigma)#, mode='constant', cval=0)
    image_back = gaussian_filter(image, sigma=sigma_back)#, mode='constant', cval=0)

    image = (image / image.max()) * 255
    image_back = (image_back / image_back.max()) * 255
    image = 0.3 * image_orig + 0.3 * image + 0.3 * image_back

    return image


def nearest_neighbor_kernel(warped_key_2_warped_val, fixed_key_2_fixed_val, scale_factor_x, scale_factor_y):
    """ Based on x- and y-coordinates of two different sets of points find a mapping between both that is based
        on the nearest neighbor.

    :param dict warped_key_2_warped_val: Warped keypoints and their x,y-coordinates stored as '[x,y]'.
    :param dict fixed_key_2_fixed_val: Fixed keypoints and their x,y-coordinates stored as '[x,y]'.
    :param float scale_factor_x: A scaling factor for the x-coordinate that is the width of the warped image
                                 space over the width of the fixed image space.
    :param float scale_factor_y: A scaling factor for the y-coordinate that is the height of the warped image
                                 space over the height of the fixed image space.
    :return: A mapping from warped_keys, that are identifiers for detected keypoints,
             to fixed_keys, that are the identifiers in the regular spaced grid and can be interpreted as classes.
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
                    (value_warped[0] - val_fix_0) ** 2 + (value_warped[1] - val_fix_1) ** 2)

                if distance < nearest_distance:
                    nearest_fixed_neighbor = key_fixed
                    nearest_distance = distance

            key_warped_2_nearest_neighbor[key_warped] = nearest_fixed_neighbor
            key_warped_2_nearest_distance[key_warped] = nearest_distance

            try:
                nearest_fixed_neighbor_2_key_warpeds[nearest_fixed_neighbor].append(key_warped)

            except KeyError:
                nearest_fixed_neighbor_2_key_warpeds[nearest_fixed_neighbor] = [key_warped]

        # 1) Evaluate all found neighbors
        for nearest_fixed_neighbor, key_warpeds in nearest_fixed_neighbor_2_key_warpeds.items():
            nearest_warped_neighbor = None
            nearest_distance = math.inf

            for key_warped in key_warpeds:
                if key_warped_2_nearest_distance[key_warped] < nearest_distance:
                    nearest_distance = key_warped_2_nearest_distance[key_warped]
                    nearest_warped_neighbor = key_warped

            if nearest_warped_neighbor != None:
                _ = warped_key_2_warped_val.pop(nearest_warped_neighbor)
                _ = fixed_key_2_fixed_val.pop(nearest_fixed_neighbor)
                warped_key_2_fixed_key[nearest_warped_neighbor] = nearest_fixed_neighbor

        # 2) Determine loop criterion
        if len(warped_key_2_warped_val) == 0 or len(fixed_key_2_fixed_val) == 0 :
            is_search_finished = True

    return warped_key_2_fixed_key


def sorting_kernel(warped_key_2_fixed_key, warped_key_2_warped_val, grid_width, grid_height):
    """ Takes the assignment of warped_keys, that are the keys for globally identifying a point, to fixed_keys,
        that are the predicted keys. Based on the values of the warped state as given in warped_val the points
        are first sorted row-wise and then column-wise, both with a bubblesort algorithm.

    :param dict warped_key_2_fixed_key: A mapping from warped_keys, that are identifiers for detected keypoints,
                                   to fixed_keys, that are the identifiers in the regular spaced grid and
                                   can be interpreted as classes.
    :param dict warped_key_2_warped_val: Warped keypoints and their x,y-coordinates stored as '[x,y]'.
    :param int grid_width: The width of the laser grid, for example 18 points.
    :param int grid_height: The height of the laser grid, for example 18 points.
    :return: The sorted mapping warped_key_2_fixed_key that was input.
    :rtype: dict
    """
    # 0) Determine the inverse dictionary with keys
    fixed_key_2_warped_key = dict()

    for warped_key in warped_key_2_fixed_key.keys():
        fixed_key = warped_key_2_fixed_key[warped_key]
        fixed_key_2_warped_key[fixed_key] = warped_key

    if len(fixed_key_2_warped_key) != len(warped_key_2_fixed_key):
        raise AssertionError('The assigment of fixed to warped keys is not unique')

    # 1) row-wise bubblesort
    for grid_height_index in range(grid_height):
        warped_vals = []
        fixed_keys = []
        for grid_width_index in range(grid_width):
            indexer = grid_height_index * grid_height + grid_width_index
            fixed_key = str(indexer)

            try:
                x = warped_key_2_warped_val[fixed_key_2_warped_key[fixed_key]][0]
            except KeyError:
                continue

            warped_vals.append(x)
            fixed_keys.append(fixed_key)

        _, fixed_keys_sorted = bubblesort(warped_vals, fixed_keys)

        # Look up which sorted key belongs to which warped then reassign key.
        # Do not resort fixed_key_2_warped_key as reference needed for finding warped keys previous to sorting.
        for fixed_key, fixed_key_sorted in zip(fixed_keys, fixed_keys_sorted):
            warped_key = fixed_key_2_warped_key[fixed_key_sorted]
            warped_key_2_fixed_key[warped_key] = fixed_key

    # 2) Determine the inverse dictionary with keys again after row-wise bubblesort
    for warped_key in warped_key_2_fixed_key.keys():
        fixed_key = warped_key_2_fixed_key[warped_key]
        fixed_key_2_warped_key[fixed_key] = warped_key

    if len(fixed_key_2_warped_key) != len(warped_key_2_fixed_key):
        raise AssertionError('The assigment of fixed to warped keys is not unique')

    # 3) column-wise bubblesort
    for grid_width_index in range(grid_width):
        warped_vals = []
        fixed_keys = []
        for grid_height_index in reversed(range(grid_height)):
            indexer = grid_height_index * grid_height + grid_width_index
            fixed_key = str(indexer)

            try:
                y = warped_key_2_warped_val[fixed_key_2_warped_key[fixed_key]][1]
            except KeyError:
                continue

            warped_vals.append(y)
            fixed_keys.append(fixed_key)

        _, fixed_keys_sorted = bubblesort(warped_vals, fixed_keys)

        # Look up which sorted key belongs to which warped then reassign key.
        # Do not resort fixed_key_2_warped_key as reference needed for finding warped keys previous to sorting.
        for fixed_key, fixed_key_sorted in zip(fixed_keys, fixed_keys_sorted):
            warped_key = fixed_key_2_warped_key[fixed_key_sorted]
            warped_key_2_fixed_key[warped_key] = fixed_key

    return warped_key_2_fixed_key


def nearest_neighbor(data_path, path_fixed, scale_factor=1):
    """ Computes the nearest neighbors and returns an accuracy.

    :param str data_path: The path in which the data is stored, can be a directory or a file
    :param str path_fixed: The path in which the fixed image json is stored
    :param float scale_factor: Scales the fixed data up or down
    :return: A dictionary that maps image ids to an accuracy
    :rtype: dict
    """
    path_fixed = path_fixed + '.json'

    if os.path.isdir(data_path):
        globs = glob(data_path + os.sep + "*_w.json")
        globs = [int(path.split(os.sep)[-1].split(".")[0].split("_")[0]) for path in globs]
        image_ids = sorted(globs)

    else:
        image_ids = [int(data_path.split(os.sep)[-1].split(".")[0].split("_")[0])]

    image_id_2_warped_key_2_fixed_key = dict()
    image_id_2_warped_key_2_warped_val = dict()
    for image_id in image_ids:
        # 0) Define desired dictionary
        warped_key_2_fixed_key = dict()

        warp_path = data_path + os.sep + "{}_w.json".format(image_id) if os.path.isdir(data_path) else data_path
        with open(warp_path) as warped_file:
            warped_json = json.load(warped_file)

        warped_val = copy.deepcopy(warped_json)

        fixed_path = path_fixed
        with open(fixed_path) as fixed_file:
            fixed_json = json.load(fixed_file)

        # 1) Sort out all obsolete points
        for key, value in list(warped_json.items()):
            if value[0] < 2.0 and value[1] < 2.0:
                _ = warped_json.pop(key)

        # 2) Compute nearest neighbor
        warped_key_2_fixed_key = nearest_neighbor_kernel(warped_json, fixed_json, scale_factor, scale_factor)
        image_id_2_warped_key_2_fixed_key[image_id] = warped_key_2_fixed_key
        image_id_2_warped_key_2_warped_val[image_id] = warped_val

    return image_id_2_warped_key_2_fixed_key, image_id_2_warped_key_2_warped_val


def check_misclassification(image_id_2_warped_key_2_fixed_key):
    """ Computes the accuracy of an image, how many in total are misclassified and whether a point is
        misclassified or not.

    :param dict image_id_2_warped_key_2_fixed_key: Contains the mapping from global keys (warped) to predicted keys (fixed).
    :return: Tuple that contains the accuracy, the misclassification per image and the if a point is miscalssified.
    :rtype: tuple
    """

    image_id_2_accuracy = dict()
    image_id_2_misclassified = dict()
    image_id_2_warped_key_2_ismisclassified = dict()

    for image_id in image_id_2_warped_key_2_fixed_key.keys():
        warped_key_2_fixed_key = image_id_2_warped_key_2_fixed_key[image_id]

        warped_key_2_ismisclassified = dict()
        counter = 0
        for warped_key, fixed_key in warped_key_2_fixed_key.items():
            if warped_key != fixed_key:
                counter += 1
                warped_key_2_ismisclassified[warped_key] = 1
            else:
                warped_key_2_ismisclassified[warped_key] = 0

        number_of_predictions = len(warped_key_2_fixed_key)
        image_id_2_accuracy[image_id] = (number_of_predictions - counter) / number_of_predictions
        image_id_2_misclassified[image_id] = counter
        image_id_2_warped_key_2_ismisclassified[image_id] = warped_key_2_ismisclassified

    return image_id_2_accuracy, image_id_2_misclassified, image_id_2_warped_key_2_ismisclassified


def regular_grid_logic(image_id_2_warped_key_2_fixed_key, image_id_2_warped_key_2_warped_val, grid_width=18, grid_height=18):
    """ Takes the assignment of warped_keys, that are the keys for globally identifying a point, to fixed_keys,
        that are the predicted keys. Based on the values of the warped state as given in warped_val the points
        are first sorted row-wise and then column-wise, both with a bubblesort algorithm.

    :param dict image_id_2_warped_key_2_fixed_key: Contains the mapping from global keys (warped) to predicted keys (fixed).
    :param dict image_id_2_warped_key_2_warped_val: Contains the x-y-values of a point given its global key (warped).
    :param int grid_width: Describes the width of the grid.
    :param int grid_height: Describes the height of the grid.
    :return: A dict that is a mapping from image_id to a dictionary mapping the warped_keys to the fixed_keys.
    :rtype: dict
    """
    # Loop over all images
    image_id_2_warped_key_2_fixed_key_update = dict()
    for image_id in image_id_2_warped_key_2_fixed_key.keys():
        warped_key_2_fixed_key = image_id_2_warped_key_2_fixed_key[image_id]
        warped_key_2_warped_val = image_id_2_warped_key_2_warped_val[image_id]

        warped_key_2_fixed_key = sorting_kernel(warped_key_2_fixed_key,
                                                warped_key_2_warped_val,
                                                grid_width,
                                                grid_height)

        image_id_2_warped_key_2_fixed_key_update[image_id] = warped_key_2_fixed_key

    return image_id_2_warped_key_2_fixed_key_update


def bubblesort(list1, list2):
    """ Takes two list and sorts both in ascending order according to the values in list1 with a bubblesort algorithm.

    :param list list1: A list on whose content the sorting is based.
    :param list list2: A list that is sorted based on list1.
    :return: A tuple containing the sorted list1 and list2
    :rtype: tuple
    """
    list_length = len(list1)
    list1_sorted = copy.deepcopy(list1)
    list2_sorted = copy.deepcopy(list2)

    # Traverse all list elements, range(list_length) also okay, but one more iteration.
    for i in range(list_length - 1):
        swap_counter = 0
        for j in range(0, list_length - i - 1):
            # Check if greater and swap entries in both lists
            if list1_sorted[j] > list1_sorted[j + 1]:
                list1_sorted[j], list1_sorted[j + 1] = list1_sorted[j + 1], list1_sorted[j]
                list2_sorted[j], list2_sorted[j + 1] = list2_sorted[j + 1], list2_sorted[j]
                swap_counter += 1

        if swap_counter == 0:
            break

    return list1_sorted, list2_sorted

def h5_file_to_dict(load_file):
    """ Transfer a h5 file to a dictionary.

    :param str load_file: The string to a h5 file.
    :return: A dictionary containing the content of the h5 file
    :rtype: dict
    """
    keys_2_values = dict()
    hf = h5py.File(load_file, 'r')
    for key in hf.keys():
        keys_2_values[key] = hf.get(key)[()]
    hf.close()

    return keys_2_values

def reconstruct_film(points, camera_calibration, laser_calibration):
    pass

if __name__ == "__main__":
    pass
