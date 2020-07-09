import numpy as np
import math
import utils
import os
import json
import random
import glob

from tensorflow import keras
from tensorflow.keras.utils import Sequence

from skimage.filters.rank import gradient
from skimage.morphology import disk

from pdb import set_trace


class SIMPLESequence(Sequence):
    def __init__(self, path, path_fixed, batch_size=32, image_ids=None, preprocess_input=None, augment=False, shuffle=False,
                 width=224, height=224, grid_width=5, grid_height=5, seed=42, multi_channel='moving'):
        """ Object for fitting to a sequence of data of the SIMPLE dataset. Laser points are considered
            as labels. In augmentation a rotation is only applied if the first attempt did not rotate a keypoint out of
            the image.

        Parameters
        ----------
        path : str
            The path to the directory where .png files are stored
        path_fixed : str
            The path to the .json and .png file the fixed image
        batch_size : int, optional
            The batch size
        image_ids : list, optional
            The image ids that will be regarded for this generator
        preprocess_input : function, optional
            Function according to the used keras model
        augment : bool, optional
            Whether to augment the data or not
        shuffle : bool, optional
            Whether to shuffle indices after each epoch
        width : int, optional
            Target image width, by default 512
        height : int, optional
            Target image height, by default 512
        grid_width : int, optional
            Laser grid width, by default 18
        grid_height : int, optional
            Laser grid height, by default 18
        seed : int, optional
            A seed to be set for shuffling
        multi_channel : string, optional
            Can be used to generate more input channels. Possible options are:
            - 'moving' for single channel with moving image only
            - 'fixed' for an additional fixed image
            - 'diff' for an additional difference image
            - 'grad' for difference and gradient
        """
        random.seed(seed)

        self._path = path
        self._path_fixed = path_fixed
        self._batch_size = batch_size
        self._preprocess_input = preprocess_input
        self._augment = augment
        self._shuffle = shuffle
        self._width = width
        self._height = height
        self._grid_width = grid_width
        self._grid_height = grid_height
        self._multi_channel = multi_channel

        self._image_id_2_scaling = dict()

        self._augmenter = utils.get_augmenter(rotation=False, flip=False)
        self._image_ids = self._get_image_ids(image_ids)
        self._size = self._get_size(self._image_ids)
        self._length = self._get_length()

    def __len__(self):
        """ On call of len(...) the length of the sequence is returned, which is an integer value that
            is used as steps_per_epoch in training.
        """
        return self._length

    def __getitem__(self, index):
        """ Overrides the __getitem__ method from Sequence, with index of the currently requested batch.
        """
        index_start = index * self._batch_size

        if index == (self._length - 1):
            index_end = self._size - 1

        else:
            index_end = index_start + self._batch_size - 1

        batch_image_ids = self._image_ids[index_start:index_end+1]

        X, y = self._get_batch(batch_image_ids)

        return X, y

    def _get_batch(self, batch_image_ids):
        """ Retrieves a batch from .png files in path, where the integer interval of [index_start, index_end] defines
            the elements of the batch. X will be preprocessed according to preprocess_input.
        """
        n_data_points = len(batch_image_ids)

        if self._multi_channel == 'fixed' or self._multi_channel == 'diff':
            X = np.zeros((n_data_points, self._height, self._width, 2))

        elif self._multi_channel == 'grad':
            X = np.zeros((n_data_points, self._height, self._width, 3))

        else:
            X = np.zeros((n_data_points, self._height, self._width, 1))

        y = np.zeros((n_data_points, self._grid_width * self._grid_height, 2, 2))

        for batch_index, image_id in enumerate(batch_image_ids):
            image, image_fixed, keypoints = self._get_image_keypoints(image_id)

            if self._multi_channel == 'fixed':
                X[batch_index] = np.concatenate((image, image_fixed), axis=2)

            elif self._multi_channel == 'diff':
                image_diff = image - image_fixed
                image_diff = image_diff - image_diff.min()
                if image_diff.max() > 1e-10:
                    image_diff = 255.0 * image_diff / image_diff.max()

                X[batch_index] = np.concatenate((image, image_diff), axis=2)

            elif self._multi_channel == 'grad':
                image_diff = image - image_fixed
                image_diff = image_diff - image_diff.min()
                if image_diff.max() > 1e-10:
                    image_diff = 255.0 * image_diff / image_diff.max()

                image_grad = gradient(image[:, :, 0] / image.max(), disk(3))[:, :, np.newaxis]
                image_grad = image_grad - image_grad.min()
                image_grad = 255.0 * image_grad / image_grad.max()

                X[batch_index] = np.concatenate((image, image_diff, image_grad), axis=2)

            else:
                X[batch_index] = image

            y[batch_index] = keypoints

        if self._preprocess_input:
            X, y = self._preprocess(X, y)

        return X, y

    def _get_image_keypoints(self, image_id):
        """ Retrieves one moving image with its associated keypoints from moving and fixed image.
        """
        # Image
        path_image = os.path.join(self._path, "{}_m.png".format(image_id))
        path_image_fixed = self._path_fixed + "_f.png"

        image = keras.preprocessing.image.load_img(path_image, color_mode="grayscale")
        image_fixed = keras.preprocessing.image.load_img(path_image_fixed, color_mode="grayscale")

        scale_factor_x = self._width / image.size[0]
        scale_factor_y = self._height / image.size[1]
        scaling = np.array([scale_factor_x, scale_factor_y])
        self._image_id_2_scaling[image_id] = scaling

        image = image.resize((self._width, self._height))
        image = keras.preprocessing.image.img_to_array(image)

        image_fixed = image_fixed.resize((self._width, self._height))
        image_fixed = keras.preprocessing.image.img_to_array(image_fixed)

        # Keypoints
        keypoints = self._get_keypoints(image_id)
        keypoints_cache = np.zeros(keypoints.shape)

        for index, keypoint in enumerate(keypoints):
            length = self._width if index % 2 == 0 else self._height
            scale = scaling[0] if index % 2 == 0 else scaling[1]

            keypoint = np.clip(keypoint * scale, 0.1, (length - 0.1))
            keypoint = np.round(keypoint, 0)
            keypoints_cache[index] = keypoint

        keypoints = keypoints_cache

        if self._augment:
            image = self._run_augmentation(image)

        return image, image_fixed, keypoints

    def _run_augmentation(self, image):
        """ Augment an image.
        """
        image = np.uint8(image)
        augmentation = self._augmenter(image=image)
        image = augmentation["image"]
        image = np.float64(image)

        return image

    def _preprocess(self, X, y):
        """ Preprocess X according to given function and scale all y values between 1 and 2.
        """
        if self._preprocess_input.__module__ == 'efficientnet.model':
            X = np.repeat(X, 3, axis=3)
            X = self._preprocess_input(X)
            X = np.expand_dims(X[:, :, :, 1], axis=3)

        else:
            X = self._preprocess_input(X)

        return X, y

    def _get_image_ids(self, image_ids):
        """ Get a list of image ids that are supposed to be in the data set.
        """
        if not image_ids:
            globs = glob.glob(self._path + os.sep + "*_m.json")
            globs = [int(path.split(os.sep)[-1].split(".")[0].split("_")[0]) for path in globs]
            image_ids = sorted(globs)

        return image_ids

    def _get_size(self, image_ids):
        """ Get the size that represents the amount of sample points.
        """
        size = len(image_ids)

        if self._batch_size > size:
            raise AssertionError('Batch size "{}" can not be larger than sample size "{}"'.format(self._batch_size,
                                                                                                  size))

        return size

    def _get_length(self):
        """ Get the length of the sequence.
        """
        length = math.ceil(self._size / self._batch_size)

        return length

    def _get_keypoints(self, image_id):
        """ Get the key points from the moving and fixed ".json" file.
        """
        path_moving = os.path.join(self._path, "{}_m.json".format(image_id))
        file_moving = open(path_moving)
        data_moving = json.load(file_moving)
        file_moving.close()

        path_fixed = self._path_fixed + "_f.json"
        file_fixed = open(path_fixed)
        data_fixed = json.load(file_fixed)
        file_fixed.close()

        n_points = self._grid_width * self._grid_height
        keypoints = np.zeros((n_points, 2, 2))

        for index in range(n_points):
            try:
                x_moving = data_moving[str(index)][0]
                y_moving = data_moving[str(index)][1]
                x_fixed = data_fixed[str(index)][0]
                y_fixed = data_fixed[str(index)][1]

                keypoints[index][0][0] = x_moving
                keypoints[index][1][0] = y_moving
                keypoints[index][0][1] = x_fixed
                keypoints[index][1][1] = y_fixed

            except KeyError:
                keypoints[index][0][0] = 0.0
                keypoints[index][1][0] = 0.0
                keypoints[index][0][1] = 0.0
                keypoints[index][1][1] = 0.0

        return keypoints

    def on_epoch_end(self):
        """ Prepare next epoch
        """
        if self._shuffle:
            to_shuffle = self._image_ids
            random.shuffle(to_shuffle)
            self._image_ids = to_shuffle

    @property
    def image_id_2_scaling(self):
        """ Get the scaling that was applied when loading images.
        """
        return self._image_id_2_scaling
