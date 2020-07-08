import numpy as np
import math
import utils
import os
import json
import random
import keys
import glob

from tensorflow import keras
from tensorflow.keras.utils import Sequence


class LASTENSequence(Sequence):
    def __init__(self, path, batch_size=32, image_ids=None, preprocess_input=None, augment=False, shuffle=False,
                 width=512, height=512, grid_width=18, grid_height=18, seed=42, label="mask", multi_channel="moving"):
        """ Object for fitting to a sequence of data of the LASTEN dataset. Laser points are considered
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
            Decide which label to return. Possible options are "mask", "keypoints" or "both".
        multi_channel : string, optional
            Can be used to generate more input channels. Possible options are:
            - 'moving' for single channel with moving image only
            - 'fixed' for an additional fixed image
        """
        random.seed(seed)

        self._path = path
        self._batch_size = batch_size
        self._preprocess_input = preprocess_input
        self._augment = augment
        self._shuffle = shuffle
        self._width = width
        self._height = height
        self._grid_width = grid_width
        self._grid_height = grid_height
        self._label = label

        self._image_id_2_scaling = dict()

        self._augmenter_rot = utils.get_augmenter(rotation=True)
        self._augmenter = utils.get_augmenter(rotation=False)
        self._image_ids = self._get_image_ids(image_ids)
        self._size = self._get_size(self._image_ids)
        self._length = self._get_length()

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

        X = np.zeros((n_data_points, self._height, self._width, 1))
        y_1 = np.zeros((n_data_points, self._height, self._width, 1))
        y_2 = np.zeros((n_data_points, self._grid_width * self._grid_height * 2))

        for batch_index, image_id in enumerate(batch_image_ids):
            image, mask, keypoints = self._get_image_mask_keypoints(image_id)

            X[batch_index] = image
            y_1[batch_index] = mask
            y_2[batch_index] = keypoints

        if self._preprocess_input:
            X, y_2 = self._preprocess(X, y_2)

        if self._label == "mask":
            y = y_1

        elif self._label == "keypoints":
            y = y_2

        elif self._label == "both":
            y = {keys.SEGMENTATION_OUTPUT_NAME: y_1,
                 keys.KEYPOINT_OUTPUT_NAME: y_2}

        else:
            raise KeyError('"{}" is not a valid label type. Valid label types are "mask",'
                           ' "keypoints" and "both".'.format(self._label))

        return X, y

    def _get_image_mask_keypoints_get_image_mask_keypoints_get_image_mask_keypoints(self, image_id):
        """ Retrieves one image with its associated mask and keypoints
        """
        # Image
        path_image = os.path.join(self._path, "{}.png".format(image_id))
        image = keras.preprocessing.image.load_img(path_image, color_mode="grayscale")

        scale_factor_x = self._width / image.size[0]
        scale_factor_y = self._height / image.size[1]
        scaling = np.array([scale_factor_x, scale_factor_y])
        self._image_id_2_scaling[image_id] = scaling

        image = image.resize((self._width, self._height))
        image = keras.preprocessing.image.img_to_array(image)

        # Mask
        path_mask = os.path.join(self._path, "{}_m.png".format(image_id))
        mask = keras.preprocessing.image.load_img(path_mask, color_mode="grayscale")
        mask = mask.resize((self._width, self._height))
        mask = keras.preprocessing.image.img_to_array(mask)
        mask = mask / 255

        # Keypoints
        keypoints = self._get_keypoints(image_id)
        keypoints_cache = np.zeros(self._grid_width * self._grid_height * 2)

        for index, keypoint in enumerate(keypoints):
            length = self._width if index % 2 == 0 else self._height
            scale = scaling[0] if index % 2 == 0 else scaling[1]

            keypoint = np.clip(keypoint * scale, 0.1, (length - 0.1))
            # Round after augmentation, if no augmentation, round here.
            keypoint = keypoint if self._augment else np.round(keypoint, 0)
            keypoints_cache[index] = keypoint

        keypoints = keypoints_cache

        if self._augment:
            image, mask, keypoints = self._run_augmentation(image, mask, keypoints)

        return image, mask, keypoints

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

    def _run_augmentation(self, image, mask, keypoints):
        """ Augment an image, its keypoints and its mask.
        """
        image = np.uint8(image)
        keypoints = np.float32(keypoints)

        # Only rotate occuring keypoints.
        keypoints_cache = []
        rot_index_2_x_index = dict()
        rot_index = 0

        for index in range(0, self._grid_height * self._grid_width):
            x_index = index * 2

            x = keypoints[x_index]
            y = keypoints[x_index + 1]

            if x >= 1 and y >= 1:
                keypoints_cache.append((x, y))

                rot_index_2_x_index[rot_index] = x_index
                rot_index += 1

        keypoints = keypoints_cache

        # Run rotated augmentation and check if points were rotated out of the image.
        augmentation = self._augmenter_rot(image=image, mask=mask, keypoints=keypoints)

        if len(augmentation["keypoints"]) < rot_index:
            augmentation = self._augmenter(image=image, mask=mask, keypoints=keypoints)

        image = augmentation["image"]

        mask = augmentation["mask"]
        mask = np.round(mask)

        # Map back the keypoints.
        keypoints = np.zeros(self._grid_width * self._grid_height * 2)

        for rot_index, keypoint in enumerate(augmentation["keypoints"]):
            x_index = rot_index_2_x_index[rot_index]

            keypoints[x_index] = round(keypoint[0], 0)
            keypoints[x_index + 1] = round(keypoint[1], 0)

        image = np.float64(image)
        mask = np.float64(mask)  # self._get_mask_from_keypoints(keypoints)
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

        normalization = np.array([[self._width, self._height]])
        normalization = np.repeat(normalization, self._grid_height * self._grid_width, axis=0).flatten()
        y = np.apply_along_axis(lambda keypoints: (keypoints / normalization) + 1, 1, y)

        return X, y

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
        """ Create a dictionary from the 'ap.points' file that maps an image_id to anterior and posterior points.
        """
        path_labels = os.path.join(self._path, "{}.json".format(image_id))
        file_labels = open(path_labels)
        data_labels = json.load(file_labels)
        file_labels.close()

        keypoints = np.zeros(self._grid_width * self._grid_height * 2)

        for key, value in data_labels.items():
            key = int(key)
            keypoints[(key * 2) + 0] = value[0]
            keypoints[(key * 2) + 1] = value[1]

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
