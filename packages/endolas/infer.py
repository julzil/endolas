from .predictors import PredictorContainer
from . import keys
import os
import sys


# ----------------------------------------------------------------------------------------------------------------------
def run_inference(path):
    file_extension = path.split(os.sep)[-1].split('.')[-1].lower()

    predictor_container = PredictorContainer()

    if file_extension in keys.ADMISSIBLE_FILE_EXTENSIONS:
        if not os.path.isfile(path):
            raise ValueError('The file "{}" does not exist.'.format(path))

        if file_extension in keys.IMAGE_FILE_EXTENSIONS:
            predictor_container.predict_image(path)

        if file_extension in keys.VIDEO_FILE_EXTENSIONS:
            predictor_container.predict_video(path)

    elif os.path.isdir(path):
        predictor_container.predict_images(path)

    else:
        raise ValueError('The file "{}" is not a directory'
                         'or a file of the types {}.'.format(path, keys.ADMISSIBLE_FILE_EXTENSIONS))

# ----------------------------------------------------------------------------------------------------------------------
