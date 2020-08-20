import numpy as np
import math

import os
import json
import random
import glob
import imageio

from . import keys
from . import utils
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from .unet import preprocess_input
from PIL import Image

from pdb import set_trace


class MapInferSequence(Sequence):
    def __init__(self, data, from_frame, to_frame, batch_size=2, width=768, height=768):
        """ Object for fitting to a sequence of data of the LASTEN dataset. Laser points are considered
            as labels. In augmentation a rotation is only applied if the first attempt did not rotate a keypoint out of
            the image.

        Parameters
        ----------
        data : ndarray
            An numpy array containing all images to infer from with shape (images, width, height)
        from_frame : int
            The frame index where to start inference
        to_frame : int
            The frame index where to end inference
        batch_size : int, optional
            The batch size
        width : int, optional
            Target image width, by default 512
        height : int, optional
            Target image height, by default 512
        """
        self._data = data
        self._from_frame = from_frame
        self._to_frame = to_frame
        self._batch_size = batch_size
        self._width = width
        self._height = height
        self._image_id_2_scaling = {}

        self._size = self._get_size()
        self._length = self._get_length()

        self._check_input()

    def __len__(self):
        """ On call of len(...) the length of the sequence is returned, which is an integer value that
            is used as steps_per_epoch in training.
        """
        return self._length

    def __getitem__(self, index):
        """ Overrides the __getitem__ method from Sequence, with index of the currently requested batch.
            X will be preprocessed according to preprocess_input.
        """
        index_start = index * self._batch_size + self._from_frame
        if index == (self._length - 1):
            index_end = self._to_frame
        else:
            index_end = index_start + self._batch_size - 1

        X, image_ids = self._get_batch_from_data(index_start, index_end)
        X = preprocess_input(X)

        return X, image_ids

    def _get_batch_from_data(self, index_start, index_end):
        """ Retrieves a batch from the data object path.
        """
        n_data_points = index_end - index_start + 1

        X = np.zeros((n_data_points, self._height, self._width, 1))

        image_ids = []

        for batch_index, image_id in enumerate(range(index_start, index_end+1)):
            image = self._get_image_from_data(image_id)

            X[batch_index] = image
            image_ids.append(image_id)

        return X, image_ids

    def _get_image_from_data(self, image_id):
        """ Retrieves image from data.
        """
        image = self._data[image_id]
        image = Image.fromarray(image[:, :, 0])
        image = self._preprocess_image(image, image_id)

        return image

    def _preprocess_image(self, image, image_id):
        """ Preprocess the image.
        """
        scale_factor_x = self._width / image.size[0]
        scale_factor_y = self._height / image.size[1]
        scaling = np.array([scale_factor_x, scale_factor_y])
        self._image_id_2_scaling[image_id] = scaling

        image = image.resize((self._width, self._height))
        image = keras.preprocessing.image.img_to_array(image)

        return image

    def _preprocess(self, X):
        """ Preprocess X according to given function and scale all y values between 1 and 2.
        """
        X = self._preprocess_input(X)

        return X

    def _get_length(self):
        """ Get the length of the sequence.
        """
        length = math.ceil(self._size / self._batch_size)

        return length

    def _get_size(self):
        """ Get the size that represents the amount of sample points.
        """
        size = self._to_frame - self._from_frame + 1

        return size

    def _check_input(self):
        """ Check if the input arguments are valid
        """
        if self._data.shape[3] != 1:
            raise ValueError('The images must be be grayscale, that means data.shape == (amount, height, width, 1)')

        if self._to_frame < 0:
            raise ValueError('"to_frame" can not be negative')

        if self._from_frame < 0:
            raise ValueError('"from_frame" can not be negative')

        if self._to_frame < self._from_frame:
            raise ValueError('"to_frame" can not be smaller "from_frame"')

        try:
            _ = self._data[self._from_frame]
            _ = self._data[self._to_frame]
        except KeyError:
            raise ValueError('"from_frame" and "to_frame" must exist in data')

        if self._batch_size < 0:
            raise ValueError('"batch_size" can not be negative')

        if self._batch_size > self._size:
            raise ValueError('"batch_size" can not be larger than size "{}"'.format(self._size))

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def image_id_2_scaling(self):
        """ Get the scaling that was applied when loading images.
        """
        return self._image_id_2_scaling
