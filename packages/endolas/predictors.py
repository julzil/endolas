from .unet import preprocess_input as pre_une
from .lastengen import LASTENSequence
from segmentation_models.losses import dice_loss
from segmentation_models.metrics import iou_score

import tensorflow as tf
import numpy as np
import json
import h5py
import os

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class _PredictorTemplate(object):
    def __init__(self, sequence, path_workdir):
        self._sequence = sequence
        self._path_workdir = path_workdir

    # To be implemented in the derived class
    def predict(self):
        raise NotImplementedError()


class SegmentationPredictor(_PredictorTemplate):
    def __init__(self, sequence, path_workdir):
        super(SegmentationPredictor, self).__init__(sequence, path_workdir)

    def predict(self):
        print("Predict Segmentation")
        dependencies = {'dice_loss': dice_loss, 'iou_score': iou_score}

        model = tf.keras.models.load_model("/home/julian/Documents/Studium/MT-Masterarbeit/Workspace/endolas/experiments/results/2_4_segmentation/weights.100.hdf5",
                                           custom_objects=dependencies)

        hf_path = os.path.join(self._path_workdir, 'segmentation.h5')
        hf = h5py.File(hf_path, 'w')

        for index, value in enumerate(self._sequence):
            image_id = self._sequence._image_ids[index]

            X, _ = value
            y_pred = model.predict(X)

            hf.create_dataset(str(image_id), data=y_pred)

        hf.close()


class PeakfindingPredictor(_PredictorTemplate):
    def __init__(self, sequence, path_workdir):
        super(PeakfindingPredictor, self).__init__(sequence, path_workdir)

    def predict(self):
        print("Predict Peakfinding")


class RegistrationPredictor(_PredictorTemplate):
    def __init__(self, sequence, path_workdir):
        super(RegistrationPredictor, self).__init__(sequence, path_workdir)

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
    def __init__(self, sequence, path_workdir):
        super(NeighborPredictor, self).__init__(sequence, path_workdir)

    def predict(self):
        print("Predict Neighbor")


class PredictorContainer(object):
    def __init__(self, path, input_type, path_workdir):
        segmenation_sequence = LASTENSequence(path, preprocess_input=pre_une, label="predict", channel="physical", input=input_type)
        peakfinding_sequence = None
        registration_sequence = None
        neighbor_sequence = None

        self._segmentation_predictor = SegmentationPredictor(segmenation_sequence, path_workdir)
        self._peakfinding_predictor = PeakfindingPredictor(peakfinding_sequence, path_workdir)
        self._registration_predictor = RegistrationPredictor(registration_sequence, path_workdir)
        self._neighbor_predictor = NeighborPredictor(neighbor_sequence, path_workdir)

        self._predictors = [self._segmentation_predictor,
                            self._peakfinding_predictor,
                            self._registration_predictor,
                            self._neighbor_predictor]

    def predict(self):
        for predictor in self._predictors:
            predictor.predict()
