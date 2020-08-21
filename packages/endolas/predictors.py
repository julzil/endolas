from .unet import preprocess_input as pre_une
from .lastengen import LASTENSequence
from .infergen import MapInferSequence
from segmentation_models.losses import dice_loss
from segmentation_models.metrics import iou_score
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
            image_id_2_prediction = self.predict_specific()

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


class SegmentationPredictor(_PredictorTemplate):
    def __init__(self, sequence, results, load_file, from_frame, to_frame, laser_maps_network):
        """ The segmentation predictor infers probability maps based on a pretrained U-Net.

        :param Sequence sequence: A tensorflow.keras.utils.Sequence that is used for prediction.
        :param dict results: The common results dictionary to write to.
        :param str load_file: A path indicating where previously computed results lie.
        :param int from_frame: The frame to start from.
        :param int to_frame: The frame to end with.
        :param str laser_maps_network: A path to the file where the trained model is present.
        """
        super(SegmentationPredictor, self).__init__(sequence, results, load_file, 'laser_maps', from_frame, to_frame)

        self._laser_maps_network = laser_maps_network

    def __str__(self):
        return "Segmentation"

    def _predict_specific(self):
        image_id_2_prediction = dict()
        dependencies = {'dice_loss': dice_loss, 'iou_score': iou_score}
        try:
            model = tf.keras.models.load_model(self._laser_maps_network, custom_objects=dependencies)
        except Exception:
            raise IOError(
                'The file "{}" could not be properly loaded as a keras model'.format(self._laser_maps_network))

        for image_id, value in enumerate(self._sequence):
            X, image_ids = value
            y_pred = model.predict(X)

            for index, image_id in enumerate(image_ids):
                image_id_2_prediction[image_id] = y_pred[index][:, :, 0]

        return image_id_2_prediction


class PeakfindingPredictor(_PredictorTemplate):
    def __init__(self, sequence, results, load_file, from_frame, to_frame, laser_peaks_sigma,
                 laser_peaks_distance, laser_peaks_threshold):
        """ A peakfinding based on the utilities from skimage is carried out.

        :param Sequence sequence: A tensorflow.keras.utils.Sequence that is used for prediction.
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
            laser_map_filtered = gaussian_filter(laser_map, sigma=self._laser_peaks_sigma)
            prediction = peak_local_max(laser_map_filtered,
                                        min_distance=self._laser_peaks_distance,
                                        threshold_abs=self._laser_peaks_threshold)
            image_id_2_prediction[image_id] = prediction

        return image_id_2_prediction


class RegistrationPredictor(_PredictorTemplate):
    def __init__(self, sequence, results, load_file, from_frame, to_frame):
        """ The registration predictor infers displacement maps based on a pretrained U-Net.

        :param Sequence sequence: A tensorflow.keras.utils.Sequence that is used for prediction.
        :param dict results: The common results dictionary to write to.
        :param load_file: A path indicating where previously computed results lie.
        :param from_frame: The frame to start from.
        :param to_frame: The frame to end with.
        """
        super(RegistrationPredictor, self).__init__(sequence, results, load_file, 'laser_displacement', from_frame, to_frame)

    def _predict_specific(self):
        pass
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


class NeighborPredictor(_PredictorTemplate):
    def __init__(self, sequence, results, load_file, from_frame, to_frame):
        """ The neighbor predictor applies displacements and carries out a nearest neighbor search.

        :param Sequence sequence: A tensorflow.keras.utils.Sequence that is used for prediction.
        :param dict results: The common results dictionary to write to.
        :param load_file: A path indicating where previously computed results lie.
        :param from_frame: The frame to start from.
        :param to_frame: The frame to end with.
        """
        super(NeighborPredictor, self).__init__(sequence, results, load_file, 'laser_nearest', from_frame, to_frame)


    def _predict_specific(self):
        pass


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
                         'laser_nearest': dict(),
                         'laser_sorted': dict()}

        self._from_frame = settings['from_frame']
        self._to_frame = settings['to_frame']

        self._predictors = []

    def build_predictors(self):
        """ Internally registers all predictors in a common container.
        """
        segmentation_sequence = MapInferSequence(self._data, self._from_frame, self._to_frame,
                                                 batch_size=self._settings['laser_maps_batch']) \
                                                 if not self._settings['load_laser_maps'] else None

        peakfinding_sequence = self._results['laser_maps'] if not self._settings['load_laser_peaks'] else None
        registration_sequence = None
        neighbor_sequence = None

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
                                                      self._to_frame))
        self._predictors.append(NeighborPredictor(neighbor_sequence,
                                                  self._results,
                                                  self._settings['load_laser_nearest_file'],
                                                  self._from_frame,
                                                  self._to_frame))

    def predict(self):
        """ Predict the results of all predictors and return result dictionary.
        """
        for predictor in self._predictors:
            predictor.predict()

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
