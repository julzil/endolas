from .predictors import PredictorContainer
from . import keys
import os
import sys


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
