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
    def __init__(self, sequence, results, load_file):
        self._sequence = sequence
        self._results = results
        self._load_file = load_file
        self._image_id_2_prediction = dict()

    # To be implemented in the derived class
    def predict(self):
        raise NotImplementedError()


class SegmentationPredictor(_PredictorTemplate):
    def __init__(self, sequence, results, laser_maps_network, load_file):
        super(SegmentationPredictor, self).__init__(sequence, results, load_file)

        self._laser_maps_network = laser_maps_network

    def predict(self):
        if self._sequence:
            print("Predict Segmentation")
            dependencies = {'dice_loss': dice_loss, 'iou_score': iou_score}
            try:
                model = tf.keras.models.load_model(self._laser_maps_network, custom_objects=dependencies)
            except Exception:
                raise IOError('The file "{}" could not be properly loaded as a keras model'.format(self._laser_maps_network))

            for image_id, value in enumerate(self._sequence):
                X, image_ids = value
                y_pred = model.predict(X)

                for index, image_id in enumerate(image_ids):
                    self._image_id_2_prediction[image_id] = y_pred[index][:, :, 0]

        else:
            print("Load Segmentation")
            try:
                self._image_id_2_prediction = h5_file_to_dict(self._load_file)
            except Exception:
                raise IOError('The file "{}" does not exist or is not a valid .h5 file'.format(self._load_file))

        self._results['laser_maps'] = self._image_id_2_prediction


class PeakfindingPredictor(_PredictorTemplate):
    def __init__(self, sequence, results, load_file):
        super(PeakfindingPredictor, self).__init__(sequence, results, load_file)

    def predict(self):
        if self._sequence:
            print("Predict Peakfinding")
            for image_id, laser_map in self._sequence['laser_maps'].items():

                laser_map_filtered = gaussian_filter(laser_map, sigma=2)
                prediction = peak_local_max(laser_map_filtered, min_distance=2, threshold_abs=0.1)
                self._image_id_2_prediction[image_id] = prediction

        else:
            print("Load Peakfinding")

        self._results['laser_peaks'] = self._image_id_2_prediction


class RegistrationPredictor(_PredictorTemplate):
    def __init__(self, sequence, results, load_file):
        super(RegistrationPredictor, self).__init__(sequence, results, load_file)

    def predict(self):
        print("Predict Registration")
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
    def __init__(self, sequence, results, load_file):
        super(NeighborPredictor, self).__init__(sequence, results, load_file)

    def predict(self):
        print("Predict Neighbor")


class PredictorContainer(object):
    def __init__(self, data, settings):
        self._data = data
        self._settings = settings
        self._results = {'laser_maps': None,
                         'laser_peaks': None,
                         'laser_displacement': None,
                         'laser_nearest': None,
                         'laser_sorted': None}

        self._from_frame = settings['from_frame']
        self._to_frame = settings['to_frame']

        self._predictors = []

    def build_predictors(self):
        segmentation_sequence = MapInferSequence(self._data, self._from_frame, self._to_frame,
                                                 batch_size=self._settings['laser_maps_batch']) \
                                                 if not self._settings['load_laser_maps'] else None
        # TODO: Go on here!
        peakfinding_sequence = self._results if not self._settings['load_laser_peaks'] else None
        registration_sequence = None
        neighbor_sequence = None

        self._predictors.append(SegmentationPredictor(segmentation_sequence,
                                                      self._results,
                                                      self._settings['laser_maps_network'],
                                                      self._settings['load_laser_maps_file']))
        self._predictors.append(PeakfindingPredictor(peakfinding_sequence,
                                                     self._results,
                                                     self._settings['load_laser_peaks_file']))
        self._predictors.append(RegistrationPredictor(registration_sequence,
                                                      self._results,
                                                      self._settings['load_laser_displacement_file']))
        self._predictors.append(NeighborPredictor(neighbor_sequence,
                                                  self._results,
                                                  self._settings['load_laser_nearest_file']))

    def predict(self):
        for predictor in self._predictors:
            predictor.predict()

        set_trace()

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
