from .predictors import PredictorContainer
from . import keys
import os
import sys
import numpy as np
from .exceptions import EndolasError

from pdb import set_trace

def debug_trace():
  '''Set a tracepoint in the Python debugger that works with Qt'''
  from PyQt5.QtCore import pyqtRemoveInputHook

  from pdb import set_trace
  pyqtRemoveInputHook()
  set_trace()


SETTINGS = {
    'from_frame': 0,
    'to_frame': 1,

    'load_laser_maps': False,
    'load_laser_maps_file': '/home/julian/Documents/Studium/MT-Masterarbeit/Workspace/endolas/scripts/results/segmentation.h5',
    'laser_maps_network': '/home/julian/Documents/Studium/MT-Masterarbeit/Workspace/endolas/packages/endolas/resources/segmentation_3_2_best.hdf5',
    'laser_maps_batch': 1,

    'load_laser_peaks': False,
    'load_laser_peaks_file': '/home/julian/Documents/Studium/MT-Masterarbeit/Workspace/endolas/scripts/results/peaks.h5',
    'laser_peaks_sigma': 2,
    'laser_peaks_distance': 5,
    'laser_peaks_threshold': 0.1,

    'load_laser_displacement': False,
    'load_laser_displacement_file': '/home/julian/Documents/Studium/MT-Masterarbeit/Workspace/endolas/scripts/results/displacement.h5',
    'laser_displacement_network': '/home/julian/Documents/Studium/MT-Masterarbeit/Workspace/endolas/packages/endolas/resources/registration_8_9_60.hdf5',
    'laser_displacement_batch': 1,

    'load_laser_deformation': False,
    'load_laser_deformation_file': '/home/julian/Documents/Studium/MT-Masterarbeit/Workspace/endolas/scripts/results/deformation.h5',

    'load_laser_nearest': False,
    'load_laser_nearest_file': '/home/julian/Documents/Studium/MT-Masterarbeit/Workspace/endolas/scripts/results/neighbor.h5',

    'load_laser_sorting': False,
    'load_laser_sorting_file': '/home/julian/Documents/Studium/MT-Masterarbeit/Workspace/endolas/scripts/results/sort.h5'}


# ----------------------------------------------------------------------------------------------------------------------
def detect_laser_keypoints(data, grid_width, grid_height, settings=None, callbacks=None):
    """ Perform the prediction for all steps of the laser detection.

    :param ndarray data: The image data with shape (frames, width, height, 3)
    :param dict settings: Settings object has passed by GUI.
    :param tuple callbacks: Callbacks used by worker thread.
    :return: A result dictionary.
    :rtype: dict
    """

    #settings = SETTINGS

    data = preprocess_data(data)
    predictor_container = PredictorContainer(data, grid_width, grid_height, settings, callbacks=callbacks)
    predictor_container.build_predictors()
    predictor_container.disable_gpus()
    predictor_container.predict()
    #predictor_container.store_results()
    results = predictor_container.get_results()

    return results


def preprocess_data(data):
    """ From the RGB image take only one channel to extract data from.

    :param ndarray data: The image data with shape (frames, width, height, 3)
    :return: The image data with shape (width, height, 1) in a dictionary.
    :rtype: ndarray
    """
    data = data[:, :, :, 0]
    data = data[:, :, :, np.newaxis]

    data = {str(image_id): val for image_id, val in enumerate(data)}

    return data
