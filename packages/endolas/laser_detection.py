from .predictors import PredictorContainer
from . import keys
import os
import sys
import numpy as np

from pdb import set_trace

settings = {
    'from_frame': 4,
    'to_frame': 5,

    'load_laser_maps': True,
    'load_laser_maps_file': '/home/julian/Documents/Studium/MT-Masterarbeit/Workspace/endolas/scripts/results/segmentation.h5',
    'laser_maps_network': '/home/julian/Documents/Studium/MT-Masterarbeit/Workspace/endolas/experiments/results/3_2_segmentation/best_weights.hdf5',
    'laser_maps_batch': 1,

    'load_laser_peaks': True,
    'load_laser_peaks_file': '/home/julian/Documents/Studium/MT-Masterarbeit/Workspace/endolas/scripts/results/segmentation.h5',
    'laser_peaks_sigma': 2,
    'laser_peaks_distance': 5,
    'laser_peaks_threshold': 0.1,

    'load_laser_displacement': False,
    'load_laser_displacement_file': '...',
    'laser_displacement_network': '...select file',
    'laser_displacement_batch': 1,

    'load_laser_nearest': False,
    'load_laser_nearest_file': '...',

    'load_laser_sorting': False,
    'load_laser_sorting_file': '...'}

# ----------------------------------------------------------------------------------------------------------------------
def run_laser_detection(data, settings=None, callbacks=None):
    """ Perform the prediction for all steps of the laser detection.

    :param ndarray data: The image data with shape (frames, width, height, 3)
    :param dict settings: Settings object has passed by GUI.
    :param tuple callbacks: Callbacks used by worker thread.
    :return: A result dictionary.
    :rtype: dict
    """
    data = preprocess_data(data)
    predictor_container = PredictorContainer(data, settings)
    predictor_container.build_predictors()
    predictor_container.disable_gpus()
    predictor_container.predict()
    predictor_container.store_results()
    results = predictor_container.get_results()

    return results


def preprocess_data(data):
    """ From the RGB image take only one channel to extract data from.

    :param ndarray data: The image data with shape (frames, width, height, 3)
    :return: The image data with shape (frames, width, height, 1)
    :rtype: ndarray
    """
    data = data[:, :, :, 0]
    data = data[:, :, :, np.newaxis]

    return data