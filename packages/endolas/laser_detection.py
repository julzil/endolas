from .predictors import PredictorContainer
from . import keys
import os
import sys
import numpy as np

from pdb import set_trace


# ----------------------------------------------------------------------------------------------------------------------
def run_inference(path, path_workdir):
    file_extension = path.split(os.sep)[-1].split('.')[-1].lower()

    predictor_container = None

    if not os.path.isdir(path_workdir):
        raise ValueError('The working directory path "{}" is not a directory'.format(path_workdir))

    if file_extension in keys.ADMISSIBLE_FILE_EXTENSIONS:
        if not os.path.isfile(path):
            raise ValueError('The file "{}" does not exist.'.format(path))

        if file_extension in keys.IMAGE_FILE_EXTENSIONS:
            predictor_container = PredictorContainer(path, "img", path_workdir)

        if file_extension in keys.VIDEO_FILE_EXTENSIONS:
            predictor_container = PredictorContainer(path, "vid", path_workdir)

    elif os.path.isdir(path):
        predictor_container = PredictorContainer(path, "dir", path_workdir)

    else:
        raise ValueError('The path "{}" is not a directory'
                         'or a file of the types {}.'.format(path, keys.ADMISSIBLE_FILE_EXTENSIONS))



    predictor_container.predict()

# ----------------------------------------------------------------------------------------------------------------------
def run_laser_detection(data, settings=None, callbacks=None):
    settings ={
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
    """

    data = data[:, :, :, 0]
    data = data[:, :, :, np.newaxis]

    return data