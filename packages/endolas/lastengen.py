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
from PIL import Image

from pdb import set_trace


class LASTENSequence(Sequence):
    def __init__(self, path, path_fixed=None, batch_size=1, image_ids=None, preprocess_input=None, augment=False,
                 shuffle=False, width=768, height=768, grid_width=18, grid_height=18, seed=42, label="mask",
                 channel="physical", input="dir", lower_frame=0, upper_frame=19):
        """ Object for fitting to a sequence of data of the LASTEN dataset. Laser points are considered
            as labels. In augmentation a rotation is only applied if the first attempt did not rotate a keypoint out of
            the image.

        :param str path: The path to the directory where .png files are stored
        :param str path_fixed: The path to the .json and .png file the fixed image
        :param int batch_size: The batch size
        :param list image_ids: The image ids that will be regarded for this generator
        :param function preprocess_input: Function according to the used keras model
        :param bool augment: Whether to augment the data or not. Augmentation is only carried out when
                             label=='mask' and channel=='physical'
        :param bool shuffle: Whether to shuffle indices after each epoch
        :param int width: Target image width, by default 512
        :param int height: Target image height, by default 512
        :param int grid_width: Laser grid width, by default 18
        :param int grid_height: Laser grid height, by default 18
        :param int seed: A seed to be set for shuffling
        :param string label: Decide which label to return. Possible options are: \n
                             - 'mask' for returning a mask as labels
                             - 'keypoints' for returning keypoints as labels
                             - 'predict' for returning None as label
        :param string channel: Can be used to generate more input channels. When input is 'vid' or 'img' there is no
                               difference between 'physical' or 'moving'. Type then depends only on provided input.
                               Possible options are: \n
                               - 'physical' for single channel with physical image only
                               - 'moving' for single channel with moving image only
                               - 'moving+fixed' for an additional fixed image

        :param string input: Is needed to decide which and how data is read. Possible options are: \n
                             - 'dir' for reading all images from a directory
                             - 'img' for reading a single image from file
                             - 'vid' for reading a video from file
        :param int lower_frame: Index for identifying the lower frame to be selected from video, only when input=='vid'
        :param int upper_frame: Index for identifying the upper frame to be selected from video, only when input=='vid'
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

        self._label = label
        self._channel = channel
        self._input = input
        self._lower_frame = lower_frame
        self._upper_frame = upper_frame

        self._image_id_2_scaling = dict()

        self._augmenter = utils.get_augmenter(rotation=True)
        self._image_ids = self._get_image_ids(image_ids)
        self._size = self._get_size(self._image_ids)
        self._length = self._get_length()

        self._check_input()

        self._video = self._load_video() if self._input == "vid" else None

    def __len__(self):
        """ On call of len(...) the length of the sequence is returned, which is an integer value that
            is used as steps_per_epoch in training.
        """
        return self._length

    def __getitem__(self, index):
        """ Overrides the __getitem__ method from Sequence, with index of the currently requested batch.
            X will be preprocessed according to preprocess_input.
        """
        index_start = index * self._batch_size
        if index == (self._length - 1):
            index_end = self._size - 1
        else:
            index_end = index_start + self._batch_size - 1
        batch_image_ids = self._image_ids[index_start:index_end + 1]

        if self._input == "dir":
            X, y = self._get_batch_from_dir(batch_image_ids)

        elif self._input == "img":
            X, y = self._get_batch_from_img(batch_image_ids[0])

        else:
            X, y = self._get_batch_from_vid(batch_image_ids)

        if self._preprocess_input:
            X = self._preprocess(X)

        return X, y

    def _get_batch_from_img(self, index):
        """ Retrieves a batch from on single .png file in path.
        """
        X = np.zeros((1, self._height, self._width, 2)) if self._channel == "moving+fixed" else \
            np.zeros((1, self._height, self._width, 1))

        image = self._get_image_from_path(self._path, index)
        image_fixed = self._get_fixed()

        X[index] = np.concatenate((image, image_fixed), axis=2) if self._channel == "moving+fixed" else image

        return X, None

    def _get_batch_from_vid(self, batch_image_ids):
        """ Retrieves a batch from a video file in path.
        """
        n_data_points = len(batch_image_ids)

        X = np.zeros((n_data_points, self._height, self._width, 2)) if self._channel == "moving+fixed" else \
            np.zeros((n_data_points, self._height, self._width, 1))

        for batch_index, image_id in enumerate(batch_image_ids):
            image = self._get_image_from_video(image_id)
            fixed = self._get_fixed()

            X[batch_index] = np.concatenate((image, fixed), axis=2) if self._channel == "moving+fixed" else image

        return X, None

    def _get_batch_from_dir(self, batch_image_ids):
        """ Retrieves a batch from .png files in path, where the integer interval of [index_start, index_end] defines
            the elements of the batch. Keypoints that are no existing, that are laser points which are not visible,
            are mapped to (0, 0).
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
            if self._channel == "physical":
                path_image = os.path.join(self._path, "{}.png".format(image_id))
            else:
                path_image = os.path.join(self._path, "{}_mov.png".format(image_id))

            image = self._get_image_from_path(path_image, image_id)
            fixed = self._get_fixed()
            mask = self._get_mask(image_id)
            keypoints = self._get_keypoints(image_id)

            if self._augment and self._channel == "physical" and self._label == "mask":
                image, mask = self._run_augmentation(image, mask)

            X[batch_index] = np.concatenate((image, fixed), axis=2) if self._channel == "moving+fixed" else image

            if self._label == "mask":
                y[batch_index] = mask

            elif self._label == "keypoints":
                y[batch_index] = keypoints

        return X, y

    def _get_image_from_video(self, image_id):
        """ Retrieves image from a video.
        """
        image = self._video[image_id]
        image = Image.fromarray(image)
        image = self._preprocess_image(image, image_id)

        return image

    def _get_image_from_path(self, path, image_id):
        """ Retrieves image from a file path.
        """
        image = keras.preprocessing.image.load_img(path, color_mode="grayscale")
        image = self._preprocess_image(image, image_id)

        return image

    def _get_fixed(self):
        """ Retrieves fixed image class attribute.
        """
        if self._channel == "moving+fixed":
            path_image_fixed = self._path_fixed + ".png"
            image_fixed = keras.preprocessing.image.load_img(path_image_fixed, color_mode="grayscale")
            image_fixed = image_fixed.resize((self._width, self._height))
            image_fixed = keras.preprocessing.image.img_to_array(image_fixed)
        else:
            image_fixed = None

        return image_fixed

    def _get_mask(self, image_id):
        """ Retrieves mask according to image ID.
        """
        if self._label == "mask":
            path_mask = os.path.join(self._path, "{}_m.png".format(image_id))
            mask = keras.preprocessing.image.load_img(path_mask, color_mode="grayscale")
            mask = mask.resize((self._width, self._height))
            mask = keras.preprocessing.image.img_to_array(mask)
            mask = mask / 255

        else:
            mask = None

        return mask

    def _get_keypoints(self, image_id):
        """ Retrieves keypoints according to image ID.
        """
        if self._label == "keypoints":
            keypoints = self._get_keypoints_from_json(image_id)
            keypoints_cache = np.zeros(keypoints.shape)

            for index, keypoint in enumerate(keypoints):
                length = self._width if index % 2 == 0 else self._height
                scale = self._image_id_2_scaling[image_id][0] if index % 2 == 0 else self._image_id_2_scaling[image_id][1]

                keypoint = np.clip(keypoint * scale, 0.1, (length - 0.1))
                keypoint = np.round(keypoint, 0)
                keypoints_cache[index] = keypoint

            keypoints = keypoints_cache

        else:
            keypoints = None

        return keypoints

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

    def _preprocess_image(self, image, image_id):
        """ Preprocesses the image.
        """
        scale_factor_x = self._width / image.size[0]
        scale_factor_y = self._height / image.size[1]
        scaling = np.array([scale_factor_x, scale_factor_y])
        self._image_id_2_scaling[image_id] = scaling

        image = image.resize((self._width, self._height))
        image = keras.preprocessing.image.img_to_array(image)

        return image

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
        if self._input == "dir":
            if not image_ids:
                globs = glob.glob(self._path + os.sep + "*.json")
                globs = [int(path.split(os.sep)[-1].split(".")[0]) for path in globs]
                image_ids = sorted(globs)

        elif self._input == "img":
            image_ids = [0]

        else:
            image_ids = list(range(self._lower_frame, self._upper_frame+1))

        return image_ids

    def _get_size(self, image_ids):
        """ Get the size that represents the amount of sample points.
        """
        if self._input == "dir":
            size = len(image_ids)

        elif self._input == "img":
            size = 1

        else:
            size = self._upper_frame - self._lower_frame + 1

        return size

    def _get_length(self):
        """ Get the length of the sequence.
        """
        length = math.ceil(self._size / self._batch_size)

        return length

    def _get_keypoints_from_json(self, image_id):
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

        if self._input in ["img", "vid"] and self._label != "predict":
            raise ValueError('Input type "{}" is only valid if label is "predict"'.format(self._input))

        if type(self._lower_frame) != int or type(self._upper_frame) != int:
            raise ValueError('"lower_frame" and "upper_frame" must be of type int')

        if self._lower_frame < 0:
            raise ValueError('"lower_frame" must not be smaller than 0')

        if self._upper_frame < self._lower_frame:
            raise ValueError('"upper_frame" must not be smaller than "lower_frame"')

        if (self._channel == 'moving+fixed' or self._label == 'keypoints') and not self._path_fixed:
            raise ValueError("Please provide a path to the fixed image and fixed keypoints")

        if self._input == "dir" and not os.path.isdir(self._path):
            raise ValueError('Path "{}" does not exist.'.format(self._path))

        file_extension = self._path.split(os.sep)[-1].split('.')[-1].lower()
        if self._input == "img" and not os.path.isfile(self._path):
            raise ValueError('Path "{}" does not exist.'.format(self._path))

        if self._input == "img" and file_extension not in keys.IMAGE_FILE_EXTENSIONS:
            raise ValueError('Path "{}" is not an admissible image file. Admissible image files are'
                             '{}'.format(self._path, keys.IMAGE_FILE_EXTENSIONS))

        if self._input == "vid" and not os.path.isfile(self._path):
            raise ValueError('Path "{}" does not exist.'.format(self._path))

        if self._input == "vid" and file_extension not in keys.VIDEO_FILE_EXTENSIONS:
            raise ValueError('Path "{}" is not an admissible video file. Admissible video files are'
                             '{}'.format(self._path, keys.VIDEO_FILE_EXTENSIONS))

        if self._batch_size > self._size:
            raise ValueError('Batch size "{}" can not be larger than sample size "{}"'.format(self._batch_size,
                                                                                              self._size))

    def _load_video(self):
        """ Loads a video from file.
        """
        vid = imageio.get_reader(self._path, 'ffmpeg')
        data = [im[:, :, 0] for im in vid.iter_data()]
        video = data

        return video

    # ------------------------------------------------------------------------------------------------------------------
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
