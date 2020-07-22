import numpy as np
import math
import utils
import os
import json
import random
import glob

from tensorflow import keras
from tensorflow.keras.utils import Sequence


class LASTENSequence(Sequence):
    def __init__(self, path, path_fixed=None, batch_size=32, image_ids=None, preprocess_input=None, augment=False,
                 shuffle=False, width=512, height=512, grid_width=18, grid_height=18, seed=42, label="mask",
                 channel="physical", input="dir"):
        """ Object for fitting to a sequence of data of the LASTEN dataset. Laser points are considered
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
            Whether to augment the data or not.
            Augmentation is only carried out when label=='mask' and channel=='physical'
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
        label : string, optional
            Decide which label to return. Possible options are:
            - 'mask' for returning a mask as labels
            - 'keypoints' for returning keypoints as labels
            - 'predict' for returning None as label
        channel : string, optional
            Can be used to generate more input channels. Possible options are:
            - 'physical' for single channel with physical image only
            - 'moving' for single channel with moving image only
            - 'moving+fixed' for an additional fixed image
        input : string, optional
            Is needed to decide which and how data is read. Possible options are:
            - 'dir' for reading all images from a directory
            - 'img' for reading a single image from file
            - 'vid' for reading a video from file
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

        if label == 'keypoints' and not path_fixed:
            raise ValueError("Please provide a path to the fixed image and keypoints")

        self._label = label
        self._channel = channel
        self._input = input

        self._image_id_2_scaling = dict()

        self._augmenter = utils.get_augmenter(rotation=True)
        self._image_ids = self._get_image_ids(image_ids)
        self._size = self._get_size(self._image_ids)
        self._length = self._get_length()

        self._check_input()

    def __len__(self):
        """ On call of len(...) the length of the sequence is returned, which is an integer value that
            is used as steps_per_epoch in training.
        """
        return self._length

    def __getitem__(self, index):
        """ Overrides the __getitem__ method from Sequence, with index of the currently requested batch
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
            the elements of the batch. X will be preprocessed according to preprocess_input, y will normalized and then
            shifted by 1 such that it is between [1,2], as mape can not be evaluated for 0. Keypoints that are
            no existing, that are laser points which are not visible, are mapped to (0, 0).
        """
        n_data_points = len(batch_image_ids)

        X = np.zeros((n_data_points, self._height, self._width, 2)) if self._channel == "moving+fixed" else \
            np.zeros((n_data_points, self._height, self._width, 1))

        if self._label == "mask":
            y = np.zeros((n_data_points, self._height, self._width, 1))

        elif self._label == "keypoints":
            y = np.zeros((n_data_points, self._grid_width * self._grid_height, 2, 2))

        else:
            y = None

        for batch_index, image_id in enumerate(batch_image_ids):
            if self._channel == "physical" and self._input != "img":
                path_image = os.path.join(self._path, "{}.png".format(image_id))
            elif self._input != "img":
                path_image = os.path.join(self._path, "{}_mov.png".format(image_id))
            else:
                path_image = self._path

            image, fixed, mask, keypoints = self._get_image_fixed_mask_keypoints(path_image, image_id)

            X[batch_index] = np.concatenate((image, fixed), axis=2) if self._channel == "moving+fixed" else image

            if self._label == "mask":
                y[batch_index] = mask

            elif self._label == "keypoints":
                y[batch_index] = keypoints

        if self._preprocess_input:
            X = self._preprocess(X)

        return X, y

    def _get_image_fixed_mask_keypoints(self, path_image, image_id):
        """ Retrieves one image with its associated fixed image, mask and keypoints
        """
        # Image
        image = keras.preprocessing.image.load_img(path_image, color_mode="grayscale")

        scale_factor_x = self._width / image.size[0]
        scale_factor_y = self._height / image.size[1]
        scaling = np.array([scale_factor_x, scale_factor_y])
        self._image_id_2_scaling[image_id] = scaling

        image = image.resize((self._width, self._height))
        image = keras.preprocessing.image.img_to_array(image)

        # Fixed Image
        if self._channel == "moving+fixed":
            path_image_fixed = self._path_fixed + ".png"
            image_fixed = keras.preprocessing.image.load_img(path_image_fixed, color_mode="grayscale")
            image_fixed = image_fixed.resize((self._width, self._height))
            image_fixed = keras.preprocessing.image.img_to_array(image_fixed)
        else:
            image_fixed = None

        # Mask
        if self._label == "mask":
            keypoints = None

            path_mask = os.path.join(self._path, "{}_m.png".format(image_id))
            mask = keras.preprocessing.image.load_img(path_mask, color_mode="grayscale")
            mask = mask.resize((self._width, self._height))
            mask = keras.preprocessing.image.img_to_array(mask)
            mask = mask / 255

        # Keypoints
        elif self._label == "keypoints":
            mask = None

            keypoints = self._get_keypoints(image_id)
            keypoints_cache = np.zeros(keypoints.shape)

            for index, keypoint in enumerate(keypoints):
                length = self._width if index % 2 == 0 else self._height
                scale = scaling[0] if index % 2 == 0 else scaling[1]

                keypoint = np.clip(keypoint * scale, 0.1, (length - 0.1))
                keypoint = np.round(keypoint, 0)
                keypoints_cache[index] = keypoint

            keypoints = keypoints_cache

        # No-Labels
        else:
            mask = None
            keypoints = None

        # Augmentation
        if self._augment and self._channel == "physical" and self._label == "mask":
            image, mask = self._run_augmentation(image, mask)

        return image, image_fixed, mask, keypoints

    def _get_mask_from_keypoints(self, keypoints):
        """ Builds the mask based on the keypoints.
        """
        mask = np.zeros((self._height, self._width))
        for index in range(0, self._grid_height * self._grid_width):
            x_index = index * 2

            # Values are already rounded.
            x = int(keypoints[x_index])
            y = int(keypoints[x_index + 1])

            if x >= 1 and y >= 1:
                mask[y][x] = 1

        mask = np.expand_dims(mask, axis=2)

        return mask

    def _run_augmentation(self, image, mask):
        """ Augment an image and its mask.
        """
        image = np.uint8(image)
        augmentation = self._augmenter(image=image, mask=mask)

        image = augmentation["image"]
        mask = augmentation["mask"]
        mask = np.round(mask)

        return image, mask

    def _preprocess(self, X):
        """ Preprocess X according to given function and scale all y values between 1 and 2.
        """
        if self._preprocess_input.__module__ == 'efficientnet.model':
            X = np.repeat(X, 3, axis=3)
            X = self._preprocess_input(X)
            X = np.expand_dims(X[:, :, :, 1], axis=3)

        else:
            X = self._preprocess_input(X)

        normalization = np.array([[self._width, self._height]])
        normalization = np.repeat(normalization, self._grid_height * self._grid_width, axis=0).flatten()
        #y = np.apply_along_axis(lambda keypoints: (keypoints / normalization) + 1, 1, y)

        return X

    def _get_image_ids(self, image_ids):
        """ Get a list of image ids that are supposed to be in the data set.
        """
        if not image_ids:
            globs = glob.glob(self._path + os.sep + "*.json")
            globs = [int(path.split(os.sep)[-1].split(".")[0]) for path in globs]
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
        path_moving = os.path.join(self._path, "{}.json".format(image_id))
        file_moving = open(path_moving)
        data_moving = json.load(file_moving)
        file_moving.close()

        path_fixed = self._path_fixed + ".json"
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

    def _check_input(self):
        """ Check if the input arguments are valid
        """
        if self._label not in ["mask", "keypoints", "predict"]:
            raise ValueError('Label "{}" is not valid, valid labels are "mask", "keypoints" and'
                             ' "predict"'.format(self._label))

        if self._channel not in ["physical", "moving", "moving+fixed"]:
            raise ValueError('Channel "{}" is not valid, valid labels are "physical", "moving" and'
                             ' "moving+fixed"'.format(self._channel))

        if self._input not in ["dir", "img", "vid"]:
            raise ValueError('Input type "{}" is not valid valid labels are "dir", "img" and'
                             ' "vid"'.format(self._channel))

        if self._input in ["img", "vid"] and self._label in ["mask", "keypoints"]:
            raise ValueError('Input type "{}" is only valid if label is "predict"'.format(self._input))

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
