from .unet import preprocess_input as pre_une
from .lastengen import LASTENSequence
from .infergen import SegmentationInferSequence
from .infergen import RegistrationInferSequence
from .closs import EuclideanLoss
from .ccall import ProgLogger
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from .utils import h5_file_to_dict
from .utils import bubblesort
from .utils import nearest_neighbor_kernel
from .utils import sorting_kernel
from .exceptions import EndolasError

import tensorflow as tf
import numpy as np
import json
import h5py
import os
import math

from pdb import set_trace

# Constants
# NamedTuple

def debug_trace():
  '''Set a tracepoint in the Python debugger that works with Qt'''
  from PyQt5.QtCore import pyqtRemoveInputHook

  from pdb import set_trace
  pyqtRemoveInputHook()
  set_trace()

# ----------------------------------------------------------------------------------------------------------------------
# --- Private Part of the Module ---------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class _PredictorTemplate(object):
    def __init__(self, sequence, grid_width, grid_height, results, load_file, results_key,
                 from_frame, to_frame, callbacks=None):
        """ A private class that serves as base class for all Predictors.
        """
        self._sequence = sequence
        self._grid_width = grid_width
        self._grid_height = grid_height
        self._results = results
        self._load_file = load_file
        self._results_key = results_key
        self._from_frame = from_frame
        self._to_frame = to_frame

        self._callbacks = callbacks
        self._progress_callback = callbacks['progress_callback'] if callbacks else None
        self._message_callback = callbacks['message_callback'] if callbacks else None
        self._cancel_callback = callbacks['cancel_callback'] if callbacks else None

    def __str__(self):
        """ To be implemented in the derived class
        """
        raise NotImplementedError

    # To be implemented in the derived class
    def _predict_specific(self):
        """ To be implemented in the derived class
        """
        raise NotImplementedError

    def predict(self):
        """ If a sequence object is present a prediction is carried out, otherwise results are loaded from file
            only in the interval given by from and to frame.
        """
        if self._cancel_callback and self._cancel_callback():
            return

        if self._sequence:
            if self._message_callback:
                self._message_callback.emit("Predict " + str(self))
            else:
                print("Predict " + str(self))
            image_id_2_prediction = self._predict_specific()

        else:
            if self._message_callback:
                self._message_callback.emit("Load " + str(self))
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

            from_to_list = list(range(self._from_frame, self._to_frame + 1))
            image_ids = list(image_id_2_prediction.keys())
            for image_id in image_ids:
                try:
                    _ = int(image_id)
                except ValueError:
                    continue

                if int(image_id) not in from_to_list:
                    _ = image_id_2_prediction.pop(image_id)

            if self._progress_callback:
                self._progress_callback.emit(1)

        self._results[self._results_key].update(image_id_2_prediction)


class _NetworkPredictorTemplate(_PredictorTemplate):
    def __init__(self, sequence, grid_width, grid_height, results, load_file, result_key, from_frame, to_frame,
                 network, callbacks=None):
        """ A private class that serves as base class for all predictors that utilize neural networks.
        """
        super(_NetworkPredictorTemplate, self).__init__(sequence, grid_width, grid_height, results, load_file,
                                                        result_key, from_frame, to_frame, callbacks=callbacks)

        self._network = network
        self._model = None
        self._network_callbacks = [ProgLogger(self._progress_callback, self._cancel_callback)] if self._callbacks else []

    def _retrieve_metadata(self):
        """ The keras model contains metadata the needs to be extracted here.
        """
        try:
            hf = h5py.File(self._network, 'r')
            self._sequence.grid_width = hf.get('grid_width')[()]
            self._sequence.grid_height = hf.get('grid_height')[()]
            hf.close()
        except Exception:
            raise ValueError('No additional metadata is present in the keras model "{}"'.format(self._network))

        if self._sequence.grid_width != self._grid_width or self._sequence.grid_height != self._grid_height:
            raise EndolasError('The selected grid width "{}" or grid height "{}" does not correspond to loaded'
                               'or desired prediction. Please set accordingly!'.format(self._grid_width, self._grid_height))

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
    """ The segmentation predictor infers probability maps based on a pretrained U-Net. In the class-specific
        implementation of the prediction, that is _predict_specific, the returned dictionary contains predictions,
        that are probability maps stored as numpy arrays with the shape (image_width, image_height).
        The predictions are accessed with the frame specific key image_id tha is for example '0'. Furthermore
        with the keys 'width' and 'height' one can access the output image width and height. The keys
        'grid_width' and 'grid_height' allow to access the grid width and grid height the image was trained on.
        With the key 'settings' it is possible to access the settings that were used for this inference.

    :param Sequence sequence: A tensorflow.keras.utils.Sequence that is used for prediction.
    :param dict results: The common results dictionary to write to.
    :param str load_file: A path indicating where previously computed results lie.
    :param int from_frame: The frame to start from.
    :param int to_frame: The frame to end with.
    :param str network: A path to the file where the trained model is present.
    """
    def __init__(self, sequence, grid_width, grid_height, results, load_file, from_frame, to_frame, network, callbacks=None):
        super(SegmentationPredictor, self).__init__(sequence, grid_width, grid_height, results, load_file, 'laser_maps',
                                                    from_frame, to_frame, network, callbacks=callbacks)

    def __str__(self):
        """ Naming for the implemented class.
        """
        return "Segmentation"

    def _predict_specific(self):
        """ Class specific implementation of the prediction.
        """
        self._retrieve_metadata()
        image_id_2_prediction = dict()

        for X, image_ids in self._sequence:
            y_pred = self._model.predict(X, callbacks=self._network_callbacks)

            for index, image_id in enumerate(image_ids):
                image_id_2_prediction[image_id] = y_pred[index][:, :, 0]

        image_id_2_prediction['width'] = self._sequence.width
        image_id_2_prediction['height'] = self._sequence.height

        image_id_2_prediction['grid_width'] = self._sequence.grid_width
        image_id_2_prediction['grid_height'] = self._sequence.grid_height

        image_id_2_prediction['settings'] = json.dumps({'segmentation_batch_box': self._sequence.batch_size,
                                                        'segmentation_network_select': self._network})

        return image_id_2_prediction


class RegistrationPredictor(_NetworkPredictorTemplate):
    """ The registration predictor infers displacement maps based on a pretrained U-Net. In the class-specific
        implementation of the prediction, that is _predict_specific, the returned dictionary contains predictions,
        that are the displacement maps, stored as numpy arrays with the shape (image_width, image_height, 2).
        The predictions are accessed with the frame specific key image_id tha is for example '0'.
        The x-displacement is stored in '[:, :, 0]', whereas the y-displacement can be found in '[:, :, 1]'.
        Furthermore with the keys 'width' and 'height' one can access the output image width and height.
        The keys 'grid_width' and 'grid_height' allow to access the grid width and grid height the image was
        trained on. The key 'fix' allows to access keypoints of the fixed image.
        With the key 'settings' it is possible to access the settings that were used for this inference.

    :param Sequence sequence: A tensorflow.keras.utils.Sequence that is used for prediction.
    :param dict results: The common results dictionary to write to.
    :param str load_file: A path indicating where previously computed results lie.
    :param int from_frame: The frame to start from.
    :param int to_frame: The frame to end with.
    :param str network: A path to the file where the trained model is present.
    """
    def __init__(self, sequence, grid_width, grid_height, results, load_file, from_frame, to_frame, network, callbacks=None):
        super(RegistrationPredictor, self).__init__(sequence, grid_width, grid_height, results, load_file,
                                                    'laser_displacement', from_frame, to_frame, network, callbacks=callbacks)

    def __str__(self):
        """ Naming for the implemented class.
        """
        return "Registration"

    def _predict_specific(self):
        """ Class specific implementation of the prediction.
        """
        self._retrieve_metadata()
        image_id_2_prediction = dict()

        for X, image_ids in self._sequence:
            y_pred = self._model.predict(X, callbacks=self._network_callbacks)

            for index, image_id in enumerate(image_ids):
                image_id_2_prediction[image_id] = y_pred[index]

        image_id_2_prediction['fix'] = json.dumps(self._sequence.fixed_index_2_xy)
        image_id_2_prediction['width'] = self._sequence.width
        image_id_2_prediction['height'] = self._sequence.height
        image_id_2_prediction['grid_width'] = self._sequence.grid_width
        image_id_2_prediction['grid_height'] = self._sequence.grid_height
        image_id_2_prediction['settings'] = json.dumps({'registration_batch_box': self._sequence.batch_size,
                                                        'registration_network_select': self._network})

        return image_id_2_prediction


class PeakfindingPredictor(_PredictorTemplate):
    """ A peakfinding based on the utilities from skimage is carried out. In the class-specific
        implementation of the prediction, that is _predict_specific, the returned string formatted dictionary
        contains predictions, that are mappings from a newly assigned key 'peaked_key' to a list of predicted
        y-coordinate and x-coordinate, where the order '[x, y]' is present. The predictions are stored as strings.
        With the key 'settings' it is possible to access the settings that were used for this inference.

    :param dict sequence: A dictionary with probability maps.
    :param dict results: The common results dictionary to write to.
    :param str load_file: A path indicating where previously computed results lie.
    :param int from_frame: The frame to start from.
    :param int to_frame: The frame to end with.
    :param float laser_peaks_sigma: The standard deviation used for priorly smoothing the image.
    :param float laser_peaks_distance: The minimal distance that peaks should have.
    :param float laser_peaks_threshold: The absolute lower intensity threshold.
    """
    def __init__(self, sequence, grid_width, grid_height, results, load_file, from_frame, to_frame, laser_peaks_sigma,
                 laser_peaks_distance, laser_peaks_threshold, callbacks=None):
        super(PeakfindingPredictor, self).__init__(sequence, grid_width, grid_height, results, load_file, 'laser_peaks',
                                                   from_frame, to_frame, callbacks=callbacks)

        self._laser_peaks_sigma = laser_peaks_sigma
        self._laser_peaks_distance = laser_peaks_distance
        self._laser_peaks_threshold = laser_peaks_threshold

    def __str__(self):
        """ Naming for the implemented class.
        """
        return "Peakfinding"

    def _predict_specific(self):
        """ Class specific implementation of the prediction.
        """
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
            prediction = {str(peaked_key): [int(val[1]), int(val[0])] for peaked_key, val in enumerate(prediction)}
            image_id_2_prediction[image_id] = json.dumps(prediction)

            if self._progress_callback:
                self._progress_callback.emit(1)

        image_id_2_prediction['settings'] = json.dumps({'peak_smoothing_box': self._laser_peaks_sigma,
                                                        'peak_dist_box': self._laser_peaks_distance,
                                                        'peak_thresh_box': self._laser_peaks_threshold})

        return image_id_2_prediction


class DeformationPredictor(_PredictorTemplate):
    """ The deformation predictor applies displacements to generate warped keypoint. The returned string formatted
        dictionary contains predictions, that are mappings from a newly assigned key 'warped_key' to a list of predicted
        x-coordinate and y-coordinate, where the order '[x, y]' is present.
        With the key 'settings' it is possible to access the settings that were used for this inference.

    :param dict sequence: A dictionary with displacement maps.
    :param dict results: The common results dictionary to write to.
    :param str load_file: A path indicating where previously computed results lie.
    :param int from_frame: The frame to start from.
    :param int to_frame: The frame to end with.
    """
    def __init__(self, sequence, grid_width, grid_height, results, load_file, from_frame, to_frame, callbacks=None):
        super(DeformationPredictor, self).__init__(sequence, grid_width, grid_height, results, load_file,
                                                   'laser_deformation', from_frame, to_frame, callbacks=callbacks)

        self._laser_peaks = self._results['laser_peaks']
        self._laser_maps = self._results['laser_maps']
        self._laser_displacement = self._results['laser_displacement']
        self._scale_factor_x = None
        self._scale_factor_y = None

    def __str__(self):
        """ Naming for the implemented class.
        """
        return "Deformation"

    def _get_scale_factors(self):
        self._scale_factor_x = self._laser_displacement['width'] / self._laser_maps['width']
        self._scale_factor_y = self._laser_displacement['height'] / self._laser_maps['height']

    def _predict_specific(self):
        """ Class specific implementation of the prediction.
        """
        image_id_2_prediction = dict()

        for image_id, disp_map in self._sequence.items():
            try:
                _ = int(image_id)
            except ValueError:
                continue

            moving_keypoints_string = self._laser_peaks[image_id]
            moving_keypoints = json.loads(moving_keypoints_string)
            warp_xy_coords = dict()

            self._get_scale_factors()

            u_x = disp_map[:, :, 0]
            u_y = disp_map[:, :, 1]
            for moving_key, moving_xy_coords in moving_keypoints.items():
                # Round here is important!
                x_coord_moving = int(round(moving_xy_coords[0] * self._scale_factor_x))
                y_coord_moving = int(round(moving_xy_coords[1] * self._scale_factor_y))

                ux = u_x[y_coord_moving][x_coord_moving]
                uy = u_y[y_coord_moving][x_coord_moving]

                x_coord_warped = int(round(x_coord_moving + ux))
                y_coord_warped = int(round(y_coord_moving + uy))

                warp_xy_coords[moving_key] = [x_coord_warped, y_coord_warped]

            prediction = warp_xy_coords
            image_id_2_prediction[image_id] = json.dumps(prediction)

            if self._progress_callback:
                self._progress_callback.emit(1)

        image_id_2_prediction['settings'] = json.dumps(dict())

        return image_id_2_prediction


class NeighborPredictor(_PredictorTemplate):
    """ The neighbor predictor carries out a nearest neighbor search.
        The returned string formatted dictionary contains mappings from 'warped_key' to 'fixed_key'.
        The predictions are accessed with the frame specific key image_id that is for example '0'.
        With the key 'settings' it is possible to access the settings that were used for this inference.

    :param dict sequence: A dictionary with displaced keypoints.
    :param dict results: The common results dictionary to write to.
    :param str load_file: A path indicating where previously computed results lie.
    :param int from_frame: The frame to start from.
    :param int to_frame: The frame to end with.
    """
    def __init__(self, sequence, grid_width, grid_height, results, load_file, from_frame, to_frame, callbacks=None):
        super(NeighborPredictor, self).__init__(sequence, grid_width, grid_height, results, load_file,
                                                'laser_nearest', from_frame, to_frame, callbacks=callbacks)

        self._laser_maps = self._results['laser_maps']
        self._laser_displacement = self._results['laser_displacement']
        self._scale_factor_x = None
        self._scale_factor_y = None

    def __str__(self):
        """ Naming for the implemented class.
        """
        return "Neighbor"

    def _get_scale_factors(self):
        """ Implementation to retrieve a scaling factor for the x-coordinate that is the width of the warped image
            space over the width of the fixed image space and a scaling factor for the y-coordinate that is the
            height of the warped image space over the height of the fixed image space.
        """
        self._scale_factor_x = self._laser_displacement['width'] / self._laser_maps['width']
        self._scale_factor_y = self._laser_displacement['height'] / self._laser_maps['height']

    def _predict_specific(self):
        """ Class specific implementation of the prediction.
        """
        self._get_scale_factors()
        image_id_2_prediction = dict()
        fixed_key_2_fixed_val = json.loads(self._laser_displacement['fix'])
        for image_id, warped_keypoints in self._sequence.items():
            try:
                _ = int(image_id)
            except ValueError:
                continue

            warped_key_2_warped_val = json.loads(warped_keypoints)
            warped_key_2_fixed_key = nearest_neighbor_kernel(warped_key_2_warped_val,
                                                             fixed_key_2_fixed_val,
                                                             self._scale_factor_x,
                                                             self._scale_factor_y)
            image_id_2_prediction[image_id] = json.dumps(warped_key_2_fixed_key)

            if self._progress_callback:
                self._progress_callback.emit(1)

        image_id_2_prediction['settings'] = json.dumps(dict())

        return image_id_2_prediction


class SortingPredictor(_PredictorTemplate):
    """ The sorting predictor uses a grid logic, that is the ascending order of the keys within the grid.
        The returned string formatted dictionary contains mappings from 'warped_key' to 'fixed_key'.
        The predictions are accessed with the frame specific key image_id that is for example '0'.
        With the key 'settings' it is possible to access the settings that were used for this inference.

    :param dict sequence: A dictionary with correspondences.
    :param dict results: The common results dictionary to write to.
    :param str load_file: A path indicating where previously computed results lie.
    :param int from_frame: The frame to start from.
    :param int to_frame: The frame to end with.
    """
    def __init__(self, sequence, grid_width, grid_height, results, load_file, from_frame, to_frame, callbacks=None):
        super(SortingPredictor, self).__init__(sequence, grid_width, grid_height, results, load_file, 'laser_sorted',
                                               from_frame, to_frame, callbacks=callbacks)

        self._laser_deformation = self._results['laser_deformation']
        self._laser_displacement = self._results['laser_displacement']

    def __str__(self):
        """ Naming for the implemented class.
        """
        return "Sorting"

    def _predict_specific(self):
        """ Class specific implementation of the prediction.
        """
        image_id_2_prediction = dict()

        grid_width = self._laser_displacement['grid_width']
        grid_height = self._laser_displacement['grid_height']

        for image_id, prediction in self._sequence.items():
            try:
                _ = int(image_id)
            except ValueError:
                continue

            warped_key_2_fixed_key = json.loads(prediction)
            warped_key_2_warped_val = json.loads(self._laser_deformation[image_id])

            warped_key_2_fixed_key = sorting_kernel(warped_key_2_fixed_key,
                                                    warped_key_2_warped_val,
                                                    grid_width,
                                                    grid_height)

            image_id_2_prediction[image_id] = json.dumps(warped_key_2_fixed_key)

            if self._progress_callback:
                self._progress_callback.emit(1)

        image_id_2_prediction['settings'] = json.dumps(dict())

        return image_id_2_prediction


class PredictorContainer(object):
    """ This container class is composed by many predictors that are all managed within.

    :param ndarray data: The image data with shape (frames, width, height, 1)
    :param dict settings: Settings object has passed by GUI.
    """
    def __init__(self, data, grid_width, grid_height, settings, callbacks=None):
        self._data = data
        self._settings = settings
        self._callbacks = callbacks
        self._grid_width = grid_width
        self._grid_height = grid_height
        self._cancel_callback = callbacks['cancel_callback'] if callbacks else None

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
        if self._cancel_callback and self._cancel_callback():
            self._results = {'laser_maps': None,
                             'laser_peaks': None,
                             'laser_displacement': None,
                             'laser_deformation': None,
                             'laser_nearest': None,
                             'laser_sorted': None}
            return

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
                                                      self._grid_width,
                                                      self._grid_height,
                                                      self._results,
                                                      self._settings['load_laser_maps_file'],
                                                      self._from_frame,
                                                      self._to_frame,
                                                      self._settings['laser_maps_network'],
                                                      callbacks=self._callbacks))
        self._predictors.append(PeakfindingPredictor(peakfinding_sequence,
                                                     self._grid_width,
                                                     self._grid_height,
                                                     self._results,
                                                     self._settings['load_laser_peaks_file'],
                                                     self._from_frame,
                                                     self._to_frame,
                                                     self._settings['laser_peaks_sigma'],
                                                     self._settings['laser_peaks_distance'],
                                                     self._settings['laser_peaks_threshold'],
                                                     callbacks=self._callbacks))
        self._predictors.append(RegistrationPredictor(registration_sequence,
                                                      self._grid_width,
                                                      self._grid_height,
                                                      self._results,
                                                      self._settings['load_laser_displacement_file'],
                                                      self._from_frame,
                                                      self._to_frame,
                                                      self._settings['laser_displacement_network'],
                                                      callbacks=self._callbacks))
        self._predictors.append(DeformationPredictor(deformation_sequence,
                                                     self._grid_width,
                                                     self._grid_height,
                                                     self._results,
                                                     self._settings['load_laser_deformation_file'],
                                                     self._from_frame,
                                                     self._to_frame,
                                                     callbacks=self._callbacks))
        self._predictors.append(NeighborPredictor(neighbor_sequence,
                                                  self._grid_width,
                                                  self._grid_height,
                                                  self._results,
                                                  self._settings['load_laser_nearest_file'],
                                                  self._from_frame,
                                                  self._to_frame,
                                                  callbacks=self._callbacks))
        self._predictors.append(SortingPredictor(sorting_sequence,
                                                 self._grid_width,
                                                 self._grid_height,
                                                 self._results,
                                                 self._settings['load_laser_sorting_file'],
                                                 self._from_frame,
                                                 self._to_frame,
                                                 callbacks=self._callbacks))

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
                hf.create_dataset(image_id, data=prediction)
            hf.close()

        if not self._settings['load_laser_peaks']:
            image_id_2_prediction = self._results['laser_peaks']
            hf_path = os.path.abspath('results/peaks.h5')
            hf = h5py.File(hf_path, 'w')
            for image_id, prediction in image_id_2_prediction.items():
                hf.create_dataset(image_id, data=prediction)
            hf.close()

        if not self._settings['load_laser_displacement']:
            image_id_2_prediction = self._results['laser_displacement']
            hf_path = os.path.abspath('results/displacement.h5')
            hf = h5py.File(hf_path, 'w')
            for image_id, prediction in image_id_2_prediction.items():
                hf.create_dataset(image_id, data=prediction)
            hf.close()

        if not self._settings['load_laser_deformation']:
            image_id_2_prediction = self._results['laser_deformation']
            hf_path = os.path.abspath('results/deformation.h5')
            hf = h5py.File(hf_path, 'w')
            for image_id, prediction in image_id_2_prediction.items():
                hf.create_dataset(image_id, data=prediction)
            hf.close()

        if not self._settings['load_laser_nearest']:
            image_id_2_prediction = self._results['laser_nearest']
            hf_path = os.path.abspath('results/neighbor.h5')
            hf = h5py.File(hf_path, 'w')
            for image_id, prediction in image_id_2_prediction.items():
                hf.create_dataset(image_id, data=prediction)
            hf.close()

        if not self._settings['load_laser_sorting']:
            image_id_2_prediction = self._results['laser_sorted']
            hf_path = os.path.abspath('results/sort.h5')
            hf = h5py.File(hf_path, 'w')
            for image_id, prediction in image_id_2_prediction.items():
                hf.create_dataset(image_id, data=prediction)
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
