from tensorflow import keras
from glob import glob
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import albumentations as albu
import seaborn as sns
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
    sns.set_style('white')
    sns.set_style('ticks')

    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['axes.labelsize'] = 6
    #plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['legend.fontsize'] = 6
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['figure.figsize'] = 2, 1.5 # 3,2
    plt.rcParams['lines.linewidth'] = 0.8
    # INFO: Default dpi = 100

def _decode_keypoint_error(y_true, y_pred, width, height):
    """ From true and predicted values find the error.
    """
    resize = np.array([width, height, width, height])

    y_true = np.apply_along_axis(lambda keypoints: (keypoints * resize), 1, (y_true - 1))
    y_pred = np.apply_along_axis(lambda keypoints: (keypoints * resize), 1, (y_pred - 1))

    y = y_pred - y_true

    return y


def _plot_keypoints_and_line(ax, labels, index, width, color, label):
    """ Help to plot a keypoint and a line.
    """
    post_x = labels[index][0]
    post_y = labels[index][1]

    ante_x = labels[index][2]
    ante_y = labels[index][3]

    ax[index].plot(post_x, post_y, color=color, marker='o', markersize=5, label=label)
    ax[index].plot(ante_x, ante_y, color=color, marker='o', markersize=5)

    x = [post_x, ante_x]
    y = [post_y, ante_y]

    fit = np.polyfit(x, y, deg=1)
    poly = np.poly1d(fit)

    post_x_ext = width
    post_y_ext = poly(post_x_ext)

    ante_x_ext = 0.0
    ante_y_ext = poly(ante_x_ext)

    ax[index].plot([post_x_ext, ante_x_ext], [post_y_ext, ante_y_ext], color=color, linewidth=1)


# ----------------------------------------------------------------------------------------------------------------------
def show(image):
    """ Display only an image.

    Parameters
    ----------
    image : PIL
        Single image to be displayed
    """
    plt.figure()
    plt.axis('off')
    plt.imshow(image, cmap='gist_yarg')
    plt.show()


def plot_midline(X, y_true, y_pred, epoch, width=256, height=512, test=False):
    """ Plots the midline on an image

    Parameters
    ----------
    X : ndarray
        The image to be displayed
    y_true : ndarray
        The true keypoint location
    y_pred : ndarray
        The predicted keypoint location
    epoch : int
        The epoch the data was evaluated for
    width : int
        Keypoint location range in x-direction
    height : int
        Keypoint location range in y-direction
    test : bool
        Whether data is based on test or validation set
    """
    _init_plot()

    resize = np.array([width, height, width, height])

    y_true = np.apply_along_axis(lambda entry: (entry - 1) * resize, 1, y_true)
    y_pred = np.apply_along_axis(lambda entry: (entry - 1) * resize, 1, y_pred)

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(9, 4.5)) #, sharex=True, sharey=True)

    for i in range(5):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].imshow(X[i, :, :, 0], cmap='gist_yarg') # 'gist_yarg'

        ax[i].set_xlim([0, width])
        ax[i].set_ylim([0, height])

        _plot_keypoints_and_line(ax, y_true, i, width, COLOR_TRUE, label='true')
        _plot_keypoints_and_line(ax, y_pred, i, width, COLOR_PREDICTION, label='prediction')

        ax[i].set_title('{}.png'.format(i))

        ax[i].legend(loc='lower left')

    save_title = 'midline_epoch_{}'.format(epoch)

    if test:
        save_title += '_test'

    else:
        save_title += '_val'

    plt.savefig(save_title + '.png', format='png', bbox_inches='tight')
    plt.savefig(save_title + '.svg', format='svg', bbox_inches='tight')


def plot_convergence(paths, series, epochs=300, sigma=3, append='', plot1=-1, plot2=0, ylabel='MAPE', log=False, lower_limit=0.0, upper_limit=1.0):
    """ Create a convergence plot.

    Parameters
    ----------
    paths : dictionary
        All experiment ids mapped to corresponding path including data
    series : int
        The series which was selected
    epochs : int
        Number of epochs to plot
    sigma : float
        Parameter for convolution filter
    normalize : bool
        Whether to normalize the data or not
    test : bool
        Whether data is based on test or validation set
    plot : int
        The index of the row to be plotted
    label : str
        The label to be displayed for y axis
    log : bool
        Whether to plot as semilog or not
    upper_limit : int
        The upper limit to plot on the y-axis
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

    plt.legend(loc=(1.0, 0.0), frameon=False)

    #plt.xticks([0, 100, 200, 300])
    plt.ylim([lower_limit, upper_limit])

    save_title = 'convergence'

    save_title += '_' + str(series)
    save_title += '_' + ylabel

    for experiment_id in paths.keys():
        save_title += '_' + str(experiment_id)

    save_title += append

    plt.savefig(save_title + '.png', format='png', bbox_inches='tight', dpi=100)
    plt.savefig(save_title + '.svg', format='svg', bbox_inches='tight')


def plot_loss_convergence(paths, epochs=300, plot=-1, label='MAPE'):
    """ Create a convergence plot.

    Parameters
    ----------
    paths : dictionary
        All experiment ids mapped to corresponding path including data
    epochs : int
        Number of epochs to plot
    sigma : float
        Parameter for convolution filter
    normalize : bool
        Whether to normalize the data or not
    test : bool
        Whether data is based on test or validation set
    plot : int
        The index of the row to be plotted
    label : str
        The label to be displayed for y axis
    log : bool
        Whether to plot as semilog or not
    """
    _init_plot()

    plt.rcParams['figure.figsize'] = 6.3, 1.5
    plt.axes().set_aspect('equal')

    filter = [0, 2, 8]

    for super_index, sigma in enumerate(filter):
        ax = plt.subplot(1, len(filter), super_index + 1)

        for experiment_id in paths.keys():
            afile = open(paths[experiment_id])
            areader = csv.reader(afile, delimiter=',')

            y = np.zeros(epochs)
            x = np.zeros(epochs)

            for index, row in enumerate(areader):
                if index == 0:
                    continue

                if index == epochs + 1:
                    break

                y[index - 1] = row[plot]
                x[index - 1] = row[0]

            if sigma > 0.0:
                y = gaussian_filter1d(y, sigma, mode='nearest')
                plt.ylabel('{}, $\sigma$ = {}'.format(label, sigma))

            else:
                plt.ylabel('{}, unsmoothed'.format(label))


            plt.semilogy(x, y, label=ID_2_LABEL[experiment_id], color=ID_2_COLOR[experiment_id],
                         linestyle=ID_2_STYLE[experiment_id])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            afile.close()

        plt.axhline(1, color='grey', linestyle=':', linewidth=2)
        plt.xlabel('epoch')
        plt.xticks([0, 100, 200, 300])
        plt.ylim([0.1, 1000])
        sns.despine(offset=5, trim=True)

        #plt.tight_layout()

        if super_index == 2:
            leg = plt.legend(bbox_to_anchor=(0.6,0.5), framealpha=0.0, handlelength=0, handletextpad=0, fancybox=True)
            leg.get_frame().set_linewidth(0.0)

            for color_index, text in enumerate(leg.get_texts()):
                text.set_color(ID_2_COLOR[color_index+1])

            for item in leg.legendHandles:
                item.set_visible(False)

    plt.subplots_adjust(wspace=0.7, hspace=0.0)

    save_title = 'convergence'

    for experiment_id in paths.keys():
        save_title += '_' + str(experiment_id)

    else:
        save_title += '_val'

    save_title += '_' + label

    plt.savefig(save_title + '.png', format='png', bbox_inches='tight', dpi=300)
    plt.savefig(save_title + '.svg', format='svg', bbox_inches='tight')


def plot_distribution(y_true, y_pred, epoch, width=256, height=512, test=False):
    """ Generate a boxplot.

    Parameters
    ----------
    y_true : ndarray
        True keypoint locations
    y_pred : ndarray
        Predicted keypoint locations
    epoch : int
        The epoch to be plotted
    width : int
        Keypoint location range in x-direction
    height : int
        Keypoint location range in y-direction
    test : bool
        Whether data is based on test or validation set
    """
    y = _decode_keypoint_error(y_true, y_pred, width, height)

    plt.rcParams['figure.figsize'] = 1.7, 4
    fig, ax = plt.subplots()
    box = ax.boxplot(y, widths=0.35, whis=[5, 95], showfliers=False, patch_artist=True,
                      labels=['x', 'y', 'x', 'y'])

    box['boxes'][0].set_facecolor(COLOR_POSTERIOR)
    box['boxes'][1].set_facecolor(COLOR_POSTERIOR)

    box['medians'][0].set_color('k')
    box['medians'][1].set_color('k')

    box['boxes'][2].set_facecolor(COLOR_ANTERIOR)
    box['boxes'][3].set_facecolor(COLOR_ANTERIOR)

    box['medians'][2].set_color('k')
    box['medians'][3].set_color('k')

    ax.axhline(0, color='k', linewidth=1)
    ax.axhline(10, linestyle=':', color='k', linewidth=1)
    ax.axhline(-10, linestyle=':',  color='k', linewidth=1)

    ax.set_ylabel("error in x- and y-direction")

    plt.ylim(-40, 40)
    plt.yticks(list(range(-40, 50, 10)))

    ax.legend([box["boxes"][0], box["boxes"][2]], ['posterior', 'anterior'], loc='lower right')

    save_title = 'boxplot_5_95_epoch_{}'.format(epoch)

    if test:
        save_title += '_test'

    else:
        save_title += '_val'

    plt.savefig(save_title + '.png', format='png', bbox_inches='tight')
    plt.savefig(save_title + '.svg', format='svg', bbox_inches='tight')


def plot_error(y_trues, y_preds, epochs, width=256, height=512, test=False):
    """ Plot the error in x- and y-direction of normalized prediction over epochs

    Parameters
    ----------
    y_trues : list
        True keypoint locations
    y_preds : list
        Predicted keypoint locations
    epochs : list
        Corresponding epochs to prediction
    width : int
        Keypoint location range in x-direction
    height : int
        Keypoint location range in y-direction
    test : bool
        Whether data is based on test or validation set
    """
    if len(y_trues) != len(y_preds) and len(y_trues) != len(epochs):
        raise AssertionError("same amount of true and predicted keypoints must be available")

    n_epochs = len(epochs)

    _init_plot()

    plt.rcParams['figure.figsize'] = 9, 3
    plt.axes().set_aspect('equal')

    for index, (y_true, y_pred, epoch) in enumerate(zip(y_trues, y_preds, epochs)):
        y = _decode_keypoint_error(y_true, y_pred, width, height)

        ax = plt.subplot(2, n_epochs, index + 1, aspect=1.0)
        ax.set_title("Epoch {}".format(epoch))
        ax.spines['bottom'].set_visible(False)
        plt.plot(y[:, 0], y[:, 1], 'o', markerfacecolor=COLOR_POSTERIOR, markersize=1, markeredgewidth=0,
                 label="posterior")
        plt.plot(0, 0, 'k+', markersize=12, markeredgewidth=1.0)
        plt.xticks([])
        plt.yticks([-width/2, 0.0, width/2])
        plt.xlim(-width/2, width/2)
        plt.ylim(-height/4, height/4)

        if index == 0:
            plt.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False)
            ax.set_ylabel("error in y-direction")

        else:
            if index+1 == n_epochs:
                plt.legend(markerscale=6, bbox_to_anchor=(1, 1), loc="upper left")
            plt.tick_params(axis='y', left=True, right=False, labelleft=False, labelright=False)

        ax = plt.subplot(2, n_epochs, n_epochs + index + 1, aspect=1.0)
        ax.spines['top'].set_visible(False)
        plt.plot(y[:, 2], y[:, 3], 'o', markerfacecolor=COLOR_ANTERIOR, markersize=1, markeredgewidth=0,
                 label="anterior")
        plt.plot(0, 0, 'k+', markersize=12, markeredgewidth=1.0)
        plt.xticks([-width/2, 0.0, width/2])
        plt.yticks([-height/4, 0.0, height/4])
        plt.xlim(-width/2, width/2)
        plt.ylim(-height/4, height/4)
        if index+1 == n_epochs:
            plt.tick_params(axis='y', left=False, right=True, labelleft=False, labelright=True)
            ax.set_ylabel("error in y-direction")
            ax.yaxis.set_label_position("right")

        else:
            if index == 0:
                plt.legend(markerscale=6, bbox_to_anchor=(0, 0), loc="lower right")
            plt.tick_params(axis='y', left=False, right=True, labelleft=False, labelright=False)
        if epoch == 80:
            ax.set_xlabel("error in x-direction")

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    save_title = 'error_over_epochs'

    if test:
        save_title += '_test'

    else:
        save_title += '_val'

    plt.savefig(save_title + '.png', format='png', bbox_inches='tight')
    plt.savefig(save_title + '.svg', format='svg', bbox_inches='tight')


def show_labeled_image(x_image, y_label):
    """ Display an image and its corresponding keypoints.

    Parameters
    ----------
    x_image : ndarray
        Single image to be displayed
    y_label : ndarray
        Coordinates of anterior and posterior point as [x_p, y_p, x_a, y_a] to be overlayed.
    """
    plt.figure()
    plt.axis('off')
    plt.plot(y_label[0], y_label[1], 'bo')
    plt.plot(y_label[2], y_label[3], 'bo')
    plt.imshow(x_image, cmap='gray')
    plt.show()


def get_png_files(path):
    """ Get a list with sorted image paths.

    Parameters
    ----------
    path : str
        Path to a directory where .png files are stored

    Returns
    -------
    list
        of all .png files numerically sorted that do not contain the string 'seg' in their identifier

    """
    files_cache = glob(os.path.join(path, "*.png"))
    files = [i for i in files_cache if "seg" not in i.split(os.sep)[-1]]
    files_sorted = sorted(files, key=lambda file_path: int(file_path.split(os.sep)[-1][:-4]))

    for index, file_path in enumerate(files_sorted):
        file_name = file_path.split(os.sep)[-1][:-4]
        if int(file_name) != index:
            raise AssertionError("Something went wrong when assigning and ID to an image name")

    return files_sorted


def get_training_data(path_training, n_data_points, preprocess_input=None, width=256, height=512):
    """ Get the training data as X, y.

    Parameters
    ----------
    path_training : str
        Path to the stored training images and labels
    n_data_points : int
        Number of data points
    preprocess_input : function, optional
        Function according to the used keras model
    width: int, optional
        Target image width, by default 256
    height: int, optional
        Target image height, by default 512

    Returns
    -------
    tuple
        of the feature vector and labels as well as mapping for scaling
    """
    image_id_2_scaling = dict()
    X_train = np.zeros((n_data_points, height, width, 1))
    y_train = np.zeros((n_data_points, 4))

    for image_id in range(n_data_points):
        path_image = os.path.join(path_training, "{}.png".format(image_id))

        image = keras.preprocessing.image.load_img(path_image, color_mode="grayscale")
        image_id_2_scaling[image_id] = (image.size[0] / width,
                                        image.size[1] / height)

        size = (width, height)
        x = image.resize(size)
        x = keras.preprocessing.image.img_to_array(x)

        X_train[image_id] = x

    if preprocess_input:
        X_train = preprocess_input(X_train)

    path_labels = os.path.join(path_training, "ap.points")
    file_labels = open(path_labels)
    data_labels = json.load(file_labels)

    for roi in data_labels["rois"]:
        z = roi["z"]
        id = roi["id"]
        pos = roi["pos"]

        if z < n_data_points:
            y_train[z][(id * 2) + 0] = (pos[0] / image_id_2_scaling[z][0])
            y_train[z][(id * 2) + 1] = (pos[1] / image_id_2_scaling[z][1])

    return X_train, y_train, image_id_2_scaling


def get_augmenter(rotation=True, flip=True, keypoints=False):
    """ Get the augmenter as used in [1]_.

    Parameters
    ----------
    rotation : bool, optional
        Property whether to include rotation or not

    rotation : bool, optional
        Property whether to include flipping or not

    keypoints : bool, optional
        Property whether to augment keypoints or not

    Returns
    -------
    object
        a Compose object of the albumentations package

    References
    ----------

    [1] https://github.com/anki-xyz/bagls/blob/master/Utils/DataGenerator.py
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

    Parameters
    ----------
    image : ndarray
        The image to be smoothed
    sigma : float, optional
        The standard deviation for the kernel
    sigma_back : float, optional
        The standard deviation for the kernel in the background

    Returns
    -------
    ndarray
        The smoothed image
    """
    image_orig = image
    image = gaussian_filter(image, sigma=sigma)
    image_back = gaussian_filter(image, sigma=sigma_back)

    image = (image / image.max()) * 255
    image_back = (image_back / image_back.max()) * 255
    image = 0.3 * image_orig + 0.3 * image + 0.3 * image_back

    return image


def nearest_neighbor(data_path, path_fixed, scale_factor=1):
    """ Computes the nearest neighbors and returns an accuracy.

    Parameters
    ----------
    data_path : str
        The path in which the data is stored, can be a directory or a file

    path_fixed : str
        The path in which the fixed image json is stored

    scale_factor : float, optional
        Scales the fixed data up or down

    Returns
    -------
    dict
        A dictionary that maps image ids to an accuracy
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
        is_search_finished = False

        while(not is_search_finished):
            key_warped_2_nearest_neighbor = dict()
            key_warped_2_nearest_distance = dict()
            nearest_fixed_neighbor_2_key_warpeds = dict()

            for key_warped, value_warped in warped_json.items():
                nearest_fixed_neighbor = None
                nearest_distance = math.inf

                for key_fixed, value_fixed in fixed_json.items():
                    val_fix_0 = value_fixed[0] * scale_factor
                    val_fix_1 = value_fixed[1] * scale_factor

                    distance = math.sqrt((value_warped[0] - val_fix_0)**2 + (value_warped[1] - val_fix_1)**2)

                    if distance < nearest_distance:
                        nearest_fixed_neighbor = key_fixed
                        nearest_distance = distance

                key_warped_2_nearest_neighbor[key_warped] = nearest_fixed_neighbor
                key_warped_2_nearest_distance[key_warped] = nearest_distance

                try:
                    nearest_fixed_neighbor_2_key_warpeds[nearest_fixed_neighbor].append(key_warped)

                except KeyError:
                    nearest_fixed_neighbor_2_key_warpeds[nearest_fixed_neighbor] = [key_warped]

            # 3) Evaluate all found neighbors
            for nearest_fixed_neighbor, key_warpeds in nearest_fixed_neighbor_2_key_warpeds.items():
                nearest_warped_neighbor = None
                nearest_distance = math.inf

                for key_warped in key_warpeds:
                    if key_warped_2_nearest_distance[key_warped] < nearest_distance:
                        nearest_distance = key_warped_2_nearest_distance[key_warped]
                        nearest_warped_neighbor = key_warped

                if nearest_warped_neighbor != None:
                    _ = warped_json.pop(nearest_warped_neighbor)
                    _ = fixed_json.pop(nearest_fixed_neighbor)
                    warped_key_2_fixed_key[nearest_warped_neighbor] = nearest_fixed_neighbor

            # 4) Determine loop criterion
            if len(warped_json) == 0:
                is_search_finished = True

        image_id_2_warped_key_2_fixed_key[image_id] = warped_key_2_fixed_key
        image_id_2_warped_key_2_warped_val[image_id] = warped_val

    return image_id_2_warped_key_2_fixed_key, image_id_2_warped_key_2_warped_val


def check_misclassification(image_id_2_warped_key_2_fixed_key):
    """ Computes the accuracy of an image, how many in total are misclassified and whether a point is
        misclassified or not.

    Parameters
    ----------
    image_id_2_warped_key_2_fixed_key : dict
        Contains the mapping from global keys (warped) to predicted keys (fixed).

    Returns
    -------
    tuple
        that contains the accuracy, the misclassification per image and the if a point is miscalssified.
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

    Parameters
    ----------
    image_id_2_warped_key_2_fixed_key : dict
        Contains the mapping from global keys (warped) to predicted keys (fixed).

    image_id_2_warped_key_2_warped_val : dict
        Contains the x-y-values of a point given its global key (warped).

    grid_width : int, optional
        Describes the width of the grid.

    grid_height : int, optional
        Describes the height of the grid.

    Returns
    -------
    dict
        that is a mapping from image_id to a dictionary mapping the warped_keys to the fixed_keys.
    """

    # Loop over all images
    image_id_2_warped_key_2_fixed_key_update = dict()
    for image_id in image_id_2_warped_key_2_fixed_key.keys():
        warped_key_2_fixed_key = image_id_2_warped_key_2_fixed_key[image_id]
        warped_key_2_warped_val = image_id_2_warped_key_2_warped_val[image_id]

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

        image_id_2_warped_key_2_fixed_key_update[image_id] = warped_key_2_fixed_key

    return image_id_2_warped_key_2_fixed_key_update


def bubblesort(list1, list2):
    """ Takes two list and sorts both in ascending order according to the values in list1 with a bubblesort algorithm.

    Parameters
    ----------
    list1 : list
        A list on whose content the sorting is based.

    list2 : list
        A list that is sorted based on list1.

    Returns
    -------
    tuple
        containing the sorted list1 and list2
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
    """

    keys_2_values = dict()
    hf = h5py.File(load_file, 'r')
    for key in hf.keys():
        keys_2_values[key] = hf.get(key)[()]
    hf.close()

    return keys_2_values

if __name__ == "__main__":
    pass
