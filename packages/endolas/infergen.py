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

from matplotlib import pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
# --- Private Part of the Module ---------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class _InferSequenceTemplate(Sequence):
    def __init__(self, data, from_frame, to_frame, batch_size):
        """ Base object for infering based on a sequence of data.
        """
        self._data = data
        self._from_frame = from_frame
        self._to_frame = to_frame
        self._batch_size = batch_size
        self._width = 0
        self._height = 0
        self._grid_width = 0
        self._grid_height = 0
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
        if self._width == 0 or self._height == 0:
            raise AssertionError('The width and height of the sequence was not read out correctly')

        index_start = index * self._batch_size + self._from_frame
        if index == (self._length - 1):
            index_end = self._to_frame
        else:
            index_end = index_start + self._batch_size - 1

        X, image_ids = self._get_batch_from_data(index_start, index_end)
        X = preprocess_input(X)

        return X, image_ids

    def _get_batch_from_data(self, index_start, index_end):
        """ To be implemented in the derived class
        """
        raise NotImplementedError

    def _get_image_from_data(self, image_id):
        """ To be implemented in the derived class
        """
        raise NotImplementedError

    def _preprocess_image(self, image, image_id=None):
        """ Preprocess the image.
        """
        scale_factor_x = self._width / image.size[0]
        scale_factor_y = self._height / image.size[1]
        scaling = np.array([scale_factor_x, scale_factor_y])
        if image_id:
            self._image_id_2_scaling[image_id] = scaling

        image = image.resize((self._width, self._height))
        image = keras.preprocessing.image.img_to_array(image)

        return image

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
        """ To be defined in the derived class.
        """
        raise NotImplementedError

    def _check_template_input(self):
        """ Check if the input arguments are valid.
        """

        if self._to_frame < 0:
            raise ValueError('"to_frame" can not be negative')

        if self._from_frame < 0:
            raise ValueError('"from_frame" can not be negative')

        if self._to_frame < self._from_frame:
            raise ValueError('"to_frame" can not be smaller "from_frame"')

        if self._batch_size < 0:
            raise ValueError('"batch_size" can not be negative')

        if self._batch_size > self._size:
            raise ValueError('"batch_size" can not be larger than size "{}"'.format(self._size))

    @property
    def image_id_2_scaling(self):
        return self._image_id_2_scaling

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, val):
        self._width = val

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, val):
        self._height = val

    @property
    def grid_width(self):
        return self._grid_width

    @grid_width.setter
    def grid_width(self, val):
        self._grid_width = val

    @property
    def grid_height(self):
        return self._grid_height

    @grid_height.setter
    def grid_height(self, val):
        self._grid_height = val


# ----------------------------------------------------------------------------------------------------------------------
# --- Public Part of the Module ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class SegmentationInferSequence(_InferSequenceTemplate):
    def __init__(self, data, from_frame, to_frame, batch_size=2):
        super(SegmentationInferSequence, self).__init__(data, from_frame, to_frame, batch_size)
        """ Object for infering based on a sequence of data.

        :param ndarray data: An numpy array containing all images to infer from with shape (images, width, height)
        :param int from_frame: The frame index where to start inference
        :param int to_frame: The frame index where to end inference
        :param int batch_size: The batch size
        """

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
        image = self._data[str(image_id)]
        image = Image.fromarray(image[:, :, 0])
        image = self._preprocess_image(image, image_id=image_id)

        return image

    def _check_input(self):
        """ Check if the input arguments are valid.
        """
        self._check_template_input()

        try:
            _ = self._data[str(self._from_frame)]
            _ = self._data[str(self._to_frame)]
        except KeyError:
            raise ValueError('"from_frame" and "to_frame" must exist in data')


class RegistrationInferSequence(_InferSequenceTemplate):
    def __init__(self, data, from_frame, to_frame, segmentation_results, batch_size=1):
        super(RegistrationInferSequence, self).__init__(data, from_frame, to_frame, batch_size)
        """ Object for infering based on a sequence of data.

        :param dict data: A dictionary containing image_id_2_prediction, where prediction is a string formatted 
                          dictionary that contains x- and y-coordinates.
        :param int from_frame: The frame index where to start inference
        :param int to_frame: The frame index where to end inference
        :param Sequence segmentation_sequence: An instance of a segmentation sequence
        :param int batch_size: The batch size
        """
        self._segmentation_results = segmentation_results
        self._fixed_image = None
        self._fixed_index_2_xy = dict()
        self._segmentation_width = None
        self._segmentation_height = None
        self._foreground_smoothing = 2
        self._background_smoothing = 15

    def _get_batch_from_data(self, index_start, index_end):
        """ Retrieves a batch from the data object path for a moving and a fixed image.
        """
        probability_map = next(iter(self._segmentation_results.values()))

        self._segmentation_width = probability_map.shape[1]
        self._segmentation_height = probability_map.shape[0]

        n_data_points = index_end - index_start + 1
        X = np.zeros((n_data_points, self._height, self._width, 2))

        image_ids = []
        for batch_index, image_id in enumerate(range(index_start, index_end+1)):
            image = self._get_image_from_data(image_id)
            fixed_image = self._get_fixed_image()

            X[batch_index] = np.concatenate((image, fixed_image), axis=2)
            image_ids.append(image_id)

        return X, image_ids

    def _get_image_from_data(self, image_id):
        """ Retrieves image from data.
        """
        xy_coords_str = self._data[image_id]
        xy_coords = json.loads(xy_coords_str)

        moving = np.zeros((self._segmentation_height, self._segmentation_width))
        for value in xy_coords.values():
            x_val = value[0]
            y_val = value[1]

            if x_val < 0 or x_val > self._segmentation_width:
                raise AssertionError('Peakfinding detected point out of the image space, could not create moving image')

            if y_val < 0 or y_val > self._segmentation_height:
                raise AssertionError('Peakfinding detected point out of the image space, could not create moving image')

            moving[y_val][x_val] = 1

        moving = utils.apply_smoothing(moving, sigma=self._foreground_smoothing, sigma_back=self._background_smoothing)

        # TODO: Set workdir here!
        plt.imsave('moving.png', moving, cmap='gray')
        moving_image = keras.preprocessing.image.load_img('moving.png', color_mode="grayscale")
        os.remove('moving.png')

        image = self._preprocess_image(moving_image, image_id=image_id)

        return image

    def _get_fixed_image(self):
        """ Create the fixed image based on image space grid width and grid height.
        """
        if not self._fixed_image:

            spacing_x = (self._segmentation_width * (6.0 / 8.0)) / self._grid_width
            spacing_y = (self._segmentation_height * (6.0 / 8.0)) / self._grid_height

            base_offset_x = self._segmentation_width / 8.0
            base_offset_y = self._segmentation_height / 8.0

            offset_x = spacing_x / 2.0 + base_offset_x
            offset_y = spacing_y / 2.0 + base_offset_y

            fixed = np.zeros((self._segmentation_height, self._segmentation_width))
            index = 0
            for index_y in reversed(range(self._grid_height)):
                y = int(round(offset_y + (spacing_y * index_y)))

                for index_x in range(self._grid_width):
                    x = int(round(offset_x + (spacing_x * index_x)))

                    if x < 0 or x > self._segmentation_width:
                        raise AssertionError('Failed to create fixed image with keypoints out of bounds.')

                    if y < 0 or y > self._segmentation_height:
                        raise AssertionError('Failed to create fixed image with keypoints out of bounds.')

                    fixed[y][x] = 1
                    self._fixed_index_2_xy[str(index)] = [x, y]

                    index += 1

            fixed = utils.apply_smoothing(fixed, sigma=self._foreground_smoothing, sigma_back=self._background_smoothing)

            # TODO: Set workdir here!
            plt.imsave('fix.png', fixed, cmap='gray')
            fixed_image = keras.preprocessing.image.load_img('fix.png', color_mode="grayscale")
            fixed_image = fixed_image.resize((self._width, self._height))
            os.remove('fix.png')

            fixed_image = self._preprocess_image(fixed_image)
        else:
            fixed_image = self._fixed_image

        return fixed_image

    def _check_input(self):
        """ Check if the input arguments are valid.
        """
        self._check_template_input()

    @property
    def fixed_index_2_xy(self):
        return self._fixed_index_2_xy
