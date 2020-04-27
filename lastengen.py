import numpy as np
import math
import utils
import os
import json
import random
import keys

from tensorflow import keras
from tensorflow.keras.utils import Sequence


class LASTENSequence(Sequence):
    def __init__(self, path, batch_size=32, image_ids=None, preprocess_input=None, augment=False, shuffle=False,
                 width=256, height=512, seed=42):
        """ Object for fitting to a sequence of data of the BAGLS dataset. Anterior and posterior points are considered
            as labels. In augmentation a rotation is only applied if the first attempt did not rotate a keypoint out of
            the image.

        Parameters
        ----------
        path : str
            The path to the directory where .png files and the ap.points file is stored
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
            Target image width, by default 256
        height : int, optional
            Target image height, by default 512
        seed : int, optional
            A seed to be set for shuffling
        """
        random.seed(seed)

        self._path = path
        self._image_ids = image_ids
        self._batch_size = batch_size
        self._preprocess_input = preprocess_input
        self._augment = augment
        self._shuffle = shuffle
        self._width = width
        self._height = height

        self._image_id_2_scaling = dict()

        self._augmenter_rot = utils.get_augmenter(rotation=True)
        self._augmenter = utils.get_augmenter(rotation=False)
        self._size = self._get_size(image_ids)
        self._length = self._get_length()
        self._image_id_2_keypoints = self._get_image_id_2_keypoints()

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

        X, y = self._get_batch(index_start, index_end)

        return X, y

    def _get_batch(self, index_start, index_end):
        """ Retrieves a batch from .png files in path, where the integer interval of [index_start, index_end] defines
            the elements of the batch. X will be preprocessed according to preprocess_input, y will normalized and then
            shifted by 1 such that it is between [1,2], as mape can not be evaluated for 0.
        """
        n_data_points = index_end - index_start + 1

        X = np.zeros((n_data_points, self._height, self._width, 1))
        y_1 = np.zeros((n_data_points, self._height, self._width, 1))
        y_2 = np.zeros((n_data_points, 4))

        for batch_index, index in enumerate(range(index_start, index_end + 1)):
            image, mask, keypoints = self._get_image_mask_keypoints(index)

            X[batch_index] = image
            y_1[batch_index] = mask
            y_2[batch_index] = keypoints

        if self._preprocess_input:
            X, y_2 = self._preprocess(X, y_2)

        y = {keys.SEGMENTATION_OUTPUT_NAME: y_1,
             keys.KEYPOINT_OUTPUT_NAME: y_2}

        return X, y

    def _get_image_mask_keypoints(self, index):
        """ Retrieves one image with its associated mask and keypoints
        """
        image_id = self._image_ids[index]

        path_image = os.path.join(self._path, "{}.png".format(image_id))

        image = keras.preprocessing.image.load_img(path_image, color_mode="grayscale")

        scale_factor_x = self._width / image.size[0]
        scale_factor_y = self._height / image.size[1]
        scaling = np.array([scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y])
        self._image_id_2_scaling[image_id] = scaling

        image = image.resize((self._width, self._height))
        image = keras.preprocessing.image.img_to_array(image)

        path_mask = os.path.join(self._path, "{}_seg.png".format(image_id))

        mask = keras.preprocessing.image.load_img(path_mask, color_mode="grayscale")
        mask = mask.resize((self._width, self._height))
        mask = keras.preprocessing.image.img_to_array(mask)
        mask = mask / 255

        keypoints = self._image_id_2_keypoints[image_id] * scaling
        keypoints = np.array([np.clip(keypoints[0], 0.1, (self._width - 0.1)),
                              np.clip(keypoints[1], 0.1, (self._height - 0.1)),
                              np.clip(keypoints[2], 0.1, (self._width - 0.1)),
                              np.clip(keypoints[3], 0.1, (self._height - 0.1))])

        if self._augment:
            image, mask, keypoints = self._run_augmentation(image, keypoints, mask)

        return image, mask, keypoints

    def _run_augmentation(self, image, keypoints, mask):
        """ Augment an image, its keypoints and its mask.
        """
        image = np.uint8(image)
        mask = np.uint8(mask)
        keypoints = np.float32(keypoints)

        keypoints = [(keypoints[0], keypoints[1]), (keypoints[2], keypoints[3])]

        augmentation = self._augmenter_rot(image=image, mask=mask, keypoints=keypoints)

        if len(augmentation["keypoints"]) < 2:
            augmentation = self._augmenter(image=image, mask=mask, keypoints=keypoints)

        image = augmentation["image"]

        mask = augmentation["mask"]
        mask = np.round(mask)

        keypoints = np.array([augmentation["keypoints"][0][0],
                              augmentation["keypoints"][0][1],
                              augmentation["keypoints"][1][0],
                              augmentation["keypoints"][1][1]])

        image = np.float64(image)
        mask = np.float64(mask)
        keypoints = np.float64(keypoints)

        return image, mask, keypoints

    def _preprocess(self, X, y):
        """ Preprocess X according to given function and scale all y values between 1 and 2.
        """
        if self._preprocess_input.__module__ == 'efficientnet.model':
            X = np.repeat(X, 3, axis=3)
            X = self._preprocess_input(X)
            X = np.expand_dims(X[:, :, :, 1], axis=3)

        else:
            X = self._preprocess_input(X)

        normalization = np.array([self._width, self._height, self._width, self._height])
        y = np.apply_along_axis(lambda keypoints: (keypoints / normalization) + 1, 1, y)

        return X, y

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

    def _get_image_id_2_keypoints(self):
        """ Create a dictionary from the 'ap.points' file that maps an image_id to anterior and posterior points.
        """
        path_labels = os.path.join(self._path, "ap.points")
        file_labels = open(path_labels)
        data_labels = json.load(file_labels)
        file_labels.close()

        existing_image_ids = set()
        for roi in data_labels["rois"]:
            existing_image_ids.add(roi["z"])

        image_id_2_y = dict()
        for image_id in existing_image_ids:
            image_id_2_y[image_id] = np.zeros(4)

        for roi in data_labels["rois"]:
            image_id = roi["z"]
            keypoint_id = roi["id"]
            values = roi["pos"]

            image_id_2_y[image_id][(keypoint_id * 2) + 0] = values[0]
            image_id_2_y[image_id][(keypoint_id * 2) + 1] = values[1]

        return image_id_2_y

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

    @property
    def image_id_2_keypoints(self):
        """ Get a dictionary where the image id is mapped to anterior and posterior points that are stored in an
            ndarray with the convention np.array([x_p, y_p, x_a, y_a]).
        """
        return self._image_id_2_keypoints
