from .unet import preprocess_input as pre_une
from .lastengen import LASTENSequence
from .infergen import SegmentationInferSequence
from .infergen import RegistrationInferSequence
from .closs import EuclideanLoss
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from .utils import h5_file_to_dict

import tensorflow as tf
import numpy as np
import json
import h5py
import os

from pdb import set_trace

# Constants
# NamedTuple


# ----------------------------------------------------------------------------------------------------------------------
# --- Private Part of the Module ---------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class _PredictorTemplate(object):
    def __init__(self, sequence, results, load_file, results_key, from_frame, to_frame):
        self._sequence = sequence
        self._results = results
        self._load_file = load_file
        self._results_key = results_key
        self._from_frame = from_frame
        self._to_frame = to_frame

    # To be implemented in the derived class
    def __str__(self):
        raise NotImplementedError

    # To be implemented in the derived class
    def _predict_specific(self):
        raise NotImplementedError

    def predict(self):
        if self._sequence:
            print("Predict " + str(self))
            image_id_2_prediction = self._predict_specific()

        else:
            print("Load " + str(self))
            try:
                image_id_2_prediction = h5_file_to_dict(self._load_file)
            except Exception:
                raise IOError('The file "{}" does not exist or is not a valid .h5 file'.format(self._load_file))

            try:
                _ = image_id_2_prediction[str(self._from_frame)]
                _ = image_id_2_prediction[str(self._to_frame)]
            except KeyError:
                raise ValueError('The frame "{}" or "{}" does not exist in loaded data "{}"'.format(self._from_frame,
                                                                                                    self._to_frame,
                                                                                                    self._load_file))

        self._results[self._results_key].update(image_id_2_prediction)


class _NetworkPredictorTemplate(_PredictorTemplate):
    def __init__(self, sequence, results, load_file, result_key, from_frame, to_frame, network):
        super(_NetworkPredictorTemplate, self).__init__(sequence, results, load_file, result_key, from_frame, to_frame)

        self._network = network
        self._model = None

    def _retrieve_metadata(self):
        try:
            hf = h5py.File(self._network, 'r')
            self._sequence.grid_width = hf.get('grid_width')[()]
            self._sequence.grid_height = hf.get('grid_height')[()]
            hf.close()
        except Exception:
            raise ValueError('No additional metadata is present in the keras model "{}"'.format(self._network))

        try:
            self._model = tf.keras.models.load_model(self._network, compile=False)
        except Exception:
            raise IOError(
                'The file "{}" could not be properly loaded as a keras model'.format(self._network))
        try:
            self._sequence.width = self._model.layers[0].input_shape[0][1]
            self._sequence.height = self._model.layers[0].input_shape[0][2]
        except Exception:
            raise ValueError('The Keras model does not contain a well defined input layer.')


# ----------------------------------------------------------------------------------------------------------------------
# --- Public Part of the Module ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class SegmentationPredictor(_NetworkPredictorTemplate):
    def __init__(self, sequence, results, load_file, from_frame, to_frame, network):
        """ The segmentation predictor infers probability maps based on a pretrained U-Net. In the class-specific
            implementation of the prediction, that is _predict_specific, the returned dictionary contains predictions,
            that are probability maps stored as numpy arrays with the shape (image_width, image_height).

        :param Sequence sequence: A tensorflow.keras.utils.Sequence that is used for prediction.
        :param dict results: The common results dictionary to write to.
        :param str load_file: A path indicating where previously computed results lie.
        :param int from_frame: The frame to start from.
        :param int to_frame: The frame to end with.
        :param str network: A path to the file where the trained model is present.
        """
        super(SegmentationPredictor, self).__init__(sequence, results, load_file, 'laser_maps', from_frame, to_frame, network)

    def __str__(self):
        return "Segmentation"

    def _predict_specific(self):
        self._retrieve_metadata()
        image_id_2_prediction = dict()

        for X, image_ids in self._sequence:
            y_pred = self._model.predict(X)

            for index, image_id in enumerate(image_ids):
                image_id_2_prediction[image_id] = y_pred[index][:, :, 0]

        image_id_2_prediction['width'] = self._sequence.width
        image_id_2_prediction['height'] = self._sequence.height

        image_id_2_prediction['grid_width'] = self._sequence.grid_width
        image_id_2_prediction['grid_height'] = self._sequence.grid_height

        return image_id_2_prediction


class RegistrationPredictor(_NetworkPredictorTemplate):
    def __init__(self, sequence, results, load_file, from_frame, to_frame, network):
        """ The registration predictor infers displacement maps based on a pretrained U-Net. In the class-specific
            implementation of the prediction, that is _predict_specific, the returned dictionary contains predictions,
            that are the displacement maps, stored as numpy arrays with the shape (image_width, image_height, 2).
            The x-displacement is stored in '[:, :, 0]', whereas the y-displacement can be found in '[:, :, 1]'.
            Furthmore via the key 'fix' one can access the keypoints of the fixed image.

        :param Sequence sequence: A tensorflow.keras.utils.Sequence that is used for prediction.
        :param dict results: The common results dictionary to write to.
        :param load_file: A path indicating where previously computed results lie.
        :param from_frame: The frame to start from.
        :param to_frame: The frame to end with.
        :param str network: A path to the file where the trained model is present.
        """
        super(RegistrationPredictor, self).__init__(sequence, results, load_file, 'laser_displacement', from_frame, to_frame, network)

    def __str__(self):
        return "Registration"

    def _predict_specific(self):
        self._retrieve_metadata()
        image_id_2_prediction = dict()

        for X, image_ids in self._sequence:
            y_pred = self._model.predict(X)

            for index, image_id in enumerate(image_ids):
                image_id_2_prediction[image_id] = y_pred[index]

        image_id_2_prediction['fix'] = json.dumps(self._sequence.fixed_index_2_yx)
        image_id_2_prediction['width'] = self._sequence.width
        image_id_2_prediction['height'] = self._sequence.height
        image_id_2_prediction['grid_width'] = self._sequence.grid_width
        image_id_2_prediction['grid_height'] = self._sequence.grid_height

        return image_id_2_prediction


class PeakfindingPredictor(_PredictorTemplate):
    def __init__(self, sequence, results, load_file, from_frame, to_frame, laser_peaks_sigma,
                 laser_peaks_distance, laser_peaks_threshold):
        """ A peakfinding based on the utilities from skimage is carried out. In the class-specific
            implementation of the prediction, that is _predict_specific, the returned dictionary contains predictions,
            that are mappings from a newly assigned key 'peaked_key' to a list of predicted y-coordinate and
            x-coordinate, where the order '[y, x]' is present. The predictions are stored as strings.

        :param dict sequence: A dictionary with probability maps.
        :param dict results: The common results dictionary to write to.
        :param load_file: A path indicating where previously computed results lie.
        :param from_frame: The frame to start from.
        :param to_frame: The frame to end with.
        :param laser_peaks_sigma: The standard deviation used for priorly smoothing the image.
        :param laser_peaks_distance: The minimal distance that peaks should have.
        :param laser_peaks_threshold: The absolute lower intensity threshold.
        """
        super(PeakfindingPredictor, self).__init__(sequence, results, load_file, 'laser_peaks', from_frame, to_frame)

        self._laser_peaks_sigma = laser_peaks_sigma
        self._laser_peaks_distance = laser_peaks_distance
        self._laser_peaks_threshold = laser_peaks_threshold

    def __str__(self):
        return "Peakfinding"

    def _predict_specific(self):
        image_id_2_prediction = dict()
        for image_id, laser_map in self._sequence.items():
            try:
                _ = int(image_id)
            except ValueError:
                continue

            laser_map_filtered = gaussian_filter(laser_map, sigma=self._laser_peaks_sigma)
            prediction = peak_local_max(laser_map_filtered,
                                        min_distance=self._laser_peaks_distance,
                                        threshold_abs=self._laser_peaks_threshold)
            prediction = {str(peaked_key): [int(val[0]), int(val[1])] for peaked_key, val in enumerate(prediction)}

            image_id_2_prediction[image_id] = json.dumps(prediction)

        return image_id_2_prediction


class DeformationPredictor(_PredictorTemplate):
    def __init__(self, sequence, results, load_file, from_frame, to_frame):
        """ The deformation predictor applies displacements to generate warped keypoint.

        :param dict sequence: A dictionary with displacement maps.
        :param dict results: The common results dictionary to write to.
        :param load_file: A path indicating where previously computed results lie.
        :param from_frame: The frame to start from.
        :param to_frame: The frame to end with.
        """
        super(DeformationPredictor, self).__init__(sequence, results, load_file, 'laser_deformation', from_frame, to_frame)

        self._laser_peaks = self._results['laser_peaks']
        self._laser_maps = self._results['laser_maps']
        self._laser_displacement = self._results['laser_displacement']

    def __str__(self):
        return "Deformation"

    def _get_scale_factors(self):
        self._scale_factor_x = self._laser_displacement['width'] / self._laser_maps['width']
        self._scale_factor_y = self._laser_displacement['height'] / self._laser_maps['height']

    def _predict_specific(self):
        image_id_2_prediction = dict()

        for image_id, disp_map in self._sequence.items():
            try:
                _ = int(image_id)
            except ValueError:
                continue

            moving_keypoints_string = self._laser_peaks[image_id]
            moving_keypoints = json.loads(moving_keypoints_string)
            warp_yx_coords = dict()

            self._get_scale_factors()

            u_x = disp_map[:, :, 0]
            u_y = disp_map[:, :, 1]
            for moving_key, moving_yx_coords in moving_keypoints.items():
                # Round here is important!
                x_coord_moving = int(round(moving_yx_coords[1] * self._scale_factor_x))
                y_coord_moving = int(round(moving_yx_coords[0] * self._scale_factor_y))

                ux = u_x[y_coord_moving][x_coord_moving]
                uy = u_y[y_coord_moving][x_coord_moving]

                x_coord_warped = int(round(x_coord_moving + ux))
                y_coord_warped = int(round(y_coord_moving + uy))

                warp_yx_coords[moving_key] = [y_coord_warped, x_coord_warped]

            prediction = warp_yx_coords
            image_id_2_prediction[image_id] = json.dumps(prediction)
        return image_id_2_prediction


class NeighborPredictor(_PredictorTemplate):
    """
    model = tf.keras.models.load_model()

    warp_val = dict()
    for index, val in enumerate(self._sequence):
        image_id = self._sequence._image_ids[index]

        if index % 100 == 0:
            print(index)

        X, y = val

        y_pred = model.predict(X)
        u_x = y_pred[0, :, :, 0]
        u_y = y_pred[0, :, :, 1]

        for inner_index in range(0, grid_points):
            x_pos = int(y[0, inner_index, 0, 0])
            y_pos = int(y[0, inner_index, 1, 0])

            ux_field = y_pred[0, :, :, 0]
            uy_field = y_pred[0, :, :, 1]

            ux = ux_field[y_pos][x_pos]
            uy = uy_field[y_pos][x_pos]

            x_pos = int(round(x_pos + ux))
            y_pos = int(round(y_pos + uy))

            warp_val[str(inner_index)] = [x_pos, y_pos]
            with open(self._path_workdir + '/{}_w.json'.format(image_id), 'w') as fp:
                json.dump(warp_val, fp)
    """
    def __init__(self, sequence, results, load_file, from_frame, to_frame):
        """ The neighbor predictor carries out a nearest neighbor search.

        :param dict sequence: A dictionary with displaced keypoints.
        :param dict results: The common results dictionary to write to.
        :param load_file: A path indicating where previously computed results lie.
        :param from_frame: The frame to start from.
        :param to_frame: The frame to end with.
        """
        super(NeighborPredictor, self).__init__(sequence, results, load_file, 'laser_nearest', from_frame, to_frame)

    def __str__(self):
        return "Neighbor"

    def _predict_specific(self):
        image_id_2_prediction = dict()
        for image_id, disp_map in self._sequence.items():
            try:
                _ = int(image_id)
            except ValueError:
                continue

            image_id_2_prediction[image_id] = 1
        return image_id_2_prediction


class SortingPredictor(_PredictorTemplate):
    def __init__(self, sequence, results, load_file, from_frame, to_frame):
        """ The sorting predictor uses a grid logic, that is the ascending order of the keys within the grid.

        :param dict sequence: A dictionary with correspondences.
        :param dict results: The common results dictionary to write to.
        :param load_file: A path indicating where previously computed results lie.
        :param from_frame: The frame to start from.
        :param to_frame: The frame to end with.
        """
        super(SortingPredictor, self).__init__(sequence, results, load_file, 'laser_sorted', from_frame, to_frame)

    def __str__(self):
        return "Sorting"

    def _predict_specific(self):
        image_id_2_prediction = dict()
        return image_id_2_prediction


class PredictorContainer(object):
    def __init__(self, data, settings):
        """ This container class is composed by many predictors that are all managed within.

        :param ndarray data: The image data with shape (frames, width, height, 1)
        :param dict settings: Settings object has passed by GUI.
        """
        self._data = data
        self._settings = settings
        self._results = {'laser_maps': dict(),
                         'laser_peaks': dict(),
                         'laser_displacement': dict(),
                         'laser_deformation': dict(),
                         'laser_nearest': dict(),
                         'laser_sorted': dict()}

        self._from_frame = settings['from_frame']
        self._to_frame = settings['to_frame']

        self._predictors = []

    def _check_results(self):
        """ Do some checks on the results.
        """
        if self._results['laser_maps']['grid_width'] != self._results['laser_displacement']['grid_width']:
            raise ValueError('The grid widths from the trained segmentation "{}" and registration "{}" network '
                             'are not the same.'.format(self._results['laser_maps']['grid_width'],
                                                        self._results['laser_displacement']['grid_width']))

        if self._results['laser_maps']['grid_height'] != self._results['laser_displacement']['grid_height']:
            raise ValueError('The grid heights from the trained segmentation "{}" and registration "{}" network '
                             'are not the same.'.format(self._results['laser_maps']['grid_height'],
                                                        self._results['laser_displacement']['grid_height']))

    def build_predictors(self):
        """ Internally registers all predictors in a common container.
        """
        segmentation_sequence = SegmentationInferSequence(self._data, self._from_frame, self._to_frame,
                                                          batch_size=self._settings['laser_maps_batch']) \
                                                          if not self._settings['load_laser_maps'] else None

        registration_sequence = RegistrationInferSequence(self._results['laser_peaks'], self._from_frame,
                                                          self._to_frame,
                                                          self._results['laser_maps'],
                                                          batch_size=self._settings['laser_displacement_batch']) \
                                                          if not self._settings['load_laser_displacement'] else None

        peakfinding_sequence = self._results['laser_maps'] if not self._settings['load_laser_peaks'] else None
        deformation_sequence = self._results['laser_displacement'] if not self._settings['load_laser_deformation'] else None
        neighbor_sequence = self._results['laser_deformation'] if not self._settings['load_laser_nearest'] else None
        sorting_sequence = self._results['laser_nearest'] if not self._settings['load_laser_sorting'] else None

        self._predictors.append(SegmentationPredictor(segmentation_sequence,
                                                      self._results,
                                                      self._settings['load_laser_maps_file'],
                                                      self._from_frame,
                                                      self._to_frame,
                                                      self._settings['laser_maps_network']))
        self._predictors.append(PeakfindingPredictor(peakfinding_sequence,
                                                     self._results,
                                                     self._settings['load_laser_peaks_file'],
                                                     self._from_frame,
                                                     self._to_frame,
                                                     self._settings['laser_peaks_sigma'],
                                                     self._settings['laser_peaks_distance'],
                                                     self._settings['laser_peaks_threshold']))
        self._predictors.append(RegistrationPredictor(registration_sequence,
                                                      self._results,
                                                      self._settings['load_laser_displacement_file'],
                                                      self._from_frame,
                                                      self._to_frame,
                                                      self._settings['laser_displacement_network']))
        self._predictors.append(DeformationPredictor(deformation_sequence,
                                                     self._results,
                                                     self._settings['load_laser_deformation_file'],
                                                     self._from_frame,
                                                     self._to_frame))
        self._predictors.append(NeighborPredictor(neighbor_sequence,
                                                  self._results,
                                                  self._settings['load_laser_nearest_file'],
                                                  self._from_frame,
                                                  self._to_frame))
        self._predictors.append(SortingPredictor(sorting_sequence,
                                                     self._results,
                                                     self._settings['load_laser_sorting_file'],
                                                     self._from_frame,
                                                     self._to_frame))

    def predict(self):
        """ Predict the results of all predictors and return result dictionary.
        """
        for predictor in self._predictors:
            predictor.predict()

        self._check_results()

        return self._results

    def store_results(self):
        """ Intermediate purpose only. Once package is finished, results will be handled by GUI.
        """
        if not self._settings['load_laser_maps']:
            image_id_2_prediction = self._results['laser_maps']
            hf_path = os.path.abspath('results/segmentation.h5')
            hf = h5py.File(hf_path, 'w')
            for image_id, prediction in image_id_2_prediction.items():
                hf.create_dataset(str(image_id), data=prediction)
            hf.close()

        if not self._settings['load_laser_peaks']:
            image_id_2_prediction = self._results['laser_peaks']
            hf_path = os.path.abspath('results/peaks.h5')
            hf = h5py.File(hf_path, 'w')
            for image_id, prediction in image_id_2_prediction.items():
                hf.create_dataset(str(image_id), data=prediction)
            hf.close()

        if not self._settings['load_laser_displacement']:
            image_id_2_prediction = self._results['laser_displacement']
            hf_path = os.path.abspath('results/displacement.h5')
            hf = h5py.File(hf_path, 'w')
            for image_id, prediction in image_id_2_prediction.items():
                hf.create_dataset(str(image_id), data=prediction)
            hf.close()

        if not self._settings['load_laser_deformation']:
            image_id_2_prediction = self._results['laser_deformation']
            hf_path = os.path.abspath('results/deformation.h5')
            hf = h5py.File(hf_path, 'w')
            for image_id, prediction in image_id_2_prediction.items():
                hf.create_dataset(str(image_id), data=prediction)
            hf.close()

    def disable_gpus(self):
        """ Disables GPUs if present and infers on CPU.
        """
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'

    def get_results(self):
        """ Return results.
        """
        return self._results
