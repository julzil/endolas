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

COLOR_POSTERIOR = '#4cb2cf'
COLOR_ANTERIOR = '#87bf4b'
COLOR_TRUE = '#f78002'
COLOR_PREDICTION = '#2269b3'

ID_2_LABEL = {1: 'MSED, fine, 2 input',
              2: 'MAE, fine, 1 input',
              3: 'MSED',
              4: 'IoU, fine, 48 filter',
              5: '-',
              6: '-',
              7: '-',
              8: '-',
              9: '-',
              10: '-'}

ID_2_STYLE = {1: '-',
              2: '-',
              3: '-',
              4: '-',
              5: '-.',
              6: '-',
              7: '-',
              8: '--',
              9: '-.',
              10: ':'}

ID_2_COLOR = {1: COLOR_POSTERIOR,
              2: COLOR_ANTERIOR,
              3: COLOR_TRUE,
              4: COLOR_PREDICTION,
              5: 'k',
              6: 'k',
              7: 'k',
              8: 'k',
              9: 'k',
              10: 'k'}
# ----------------------------------------------------------------------------------------------------------------------
def _init_plot():
    """ Initialize all plot settings @ default dpi=100.
    """
    sns.set_style('white')
    sns.set_style('ticks')

    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['axes.labelsize'] = 8
    #plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

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


def _custom_loss(labels, prediction, loss_type='maed'):
    """ Compute the custom loss.
    """
    batch_size = 4 #labels.shape[0]
    loss = 0.0

    for batch_index in range(0, batch_size):
        ux = prediction[batch_index, :, :, 0]
        uy = prediction[batch_index, :, :, 1]

        x_mov = labels[batch_index, :, 0, 0]
        y_mov = labels[batch_index, :, 1, 0]
        x_mov_int = keras.backend.cast(x_mov, "int32")
        y_mov_int = keras.backend.cast(y_mov, "int32")

        x_fix = labels[batch_index, :, 0, 1]
        y_fix = labels[batch_index, :, 1, 1]

        ux_mov = get_displacement(ux, x_mov_int, y_mov_int)
        uy_mov = get_displacement(uy, x_mov_int, y_mov_int)

        x_squared = keras.backend.square(x_mov + ux_mov - x_fix)
        y_squared = keras.backend.square(y_mov + uy_mov - y_fix)

        sum_of_squares = x_squared + y_squared
        euclidean_distance = keras.backend.sqrt(sum_of_squares)

        if loss_type == 'maed':
            loss += keras.backend.mean(euclidean_distance)

        elif loss_type == 'msed':
            loss += keras.backend.mean(sum_of_squares)

        else:
            loss += 0.0

    loss = loss / batch_size

    return loss


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


def plot_convergence(paths, epochs=300, sigma=3, normalize=False, test=False, plot=-1, label='MAPE', log=False):
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

    if not normalize:
        plt.axhline(1, color='k', linewidth=0.5)

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
        
        if normalize:
            y = y / y[0]
            if log:
                plt.semilogy(x, y, label=ID_2_LABEL[experiment_id], color='k', linestyle=ID_2_STYLE[experiment_id])

            else:
                plt.plot(x, y, label=ID_2_LABEL[experiment_id], color='k', linestyle=ID_2_STYLE[experiment_id])

            plt.ylabel('{}, normalized'.format(label))

        else:
            if log:
                plt.semilogy(x, y, label=ID_2_LABEL[experiment_id], color=ID_2_COLOR[experiment_id],
                             linestyle=ID_2_STYLE[experiment_id])

            else:
                plt.plot(x, y, label=ID_2_LABEL[experiment_id], color=ID_2_COLOR[experiment_id],
                         linestyle=ID_2_STYLE[experiment_id])

            plt.ylabel('{}'.format(label))

        afile.close()

    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    save_title = 'convergence'

    for experiment_id in paths.keys():
        save_title += '_' + str(experiment_id)

    if normalize:
        save_title += '_norm'

    if test:
        save_title += '_test'

    else:
        save_title += '_val'

    save_title += '_' + label

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

    plt.savefig(save_title + '.png', format='png', bbox_inches='tight',dpi=300)
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


def get_augmenter(rotation=True, flip=True):
    """ Get the augmenter as used in [1]_.

    Parameters
    ----------
    rotation : bool, optional
        Property whether to include rotation or not

    rotation : bool, optional
        Property whether to include flipping or not

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

    augmenter = albu.Compose(transformers, keypoint_params=albu.KeypointParams(format='xy'), p=1)

    return augmenter


def get_displacement(u, x, y):
    """ Use the keras backend functionality to compute the displacement in a vectorized way.

    Parameters
    ----------
    u : Tensor (width, height)
        Predicted displacement field

    x : Tensor (n_keypoints)
        x-coordinate of key point position

    y : Tensor (n_keypoints)
        y-coordinate of key point position

    Returns
    -------
    Tensor (n_keypoints)
        The displacement of each keypoint

    """
    length = 25 # x.shape[0]
    indices = [val * length + val for val in range(0, length)]

    u = keras.backend.gather(u, y)
    u = keras.backend.transpose(u)
    u = keras.backend.gather(u, x)
    u = keras.backend.flatten(u)
    u = keras.backend.gather(u, indices)

    return u


def maed_loss(labels, prediction):
    """ Compute the mean absolute euclidean distance.

    Parameters
    ----------
    labels : Tensor
        The labels forwarded by the network

    prediction : Tensor
        The prediction forwarded by the network

    Returns
    -------
    float
        The loss value
    """
    return _custom_loss(labels, prediction, loss_type='maed')


def msed_loss(labels, prediction):
    """ Compute the mean absolute euclidean distance.

    Parameters
    ----------
    labels : Tensor
        The labels forwarded by the network

    prediction : Tensor
        The prediction forwarded by the network

    Returns
    -------
    float
        The loss value
    """
    return _custom_loss(labels, prediction, loss_type='msed')


def apply_smoothing(image):
    """ Smooth an image with two gaussian kernels.

    Parameters
    ----------
    image : ndarray
        The image to be smoothed

    Returns
    -------
    ndarray
        The smoothed image
    """
    sigma = 1.0
    sigma_back = 10

    image_orig = image
    image = gaussian_filter(image, sigma=sigma)
    image_back = gaussian_filter(image, sigma=sigma_back)

    image = (image / image.max()) * 255
    image_back = (image_back / image_back.max()) * 255
    image = 0.3 * image_orig + 0.3 * image + 0.3 * image_back

    return image

if __name__ == "__main__":
    pass
