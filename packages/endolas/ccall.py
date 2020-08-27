from tensorflow import keras
import csv
import time
import os


class TimeHistory(keras.callbacks.Callback):
    """ Custom callback object to store time spend for epoch in .csv file

    :param str path: A path where to store the log.
    """
    def __init__(self, path):
        super(TimeHistory, self).__init__()
        self.path = path
        self.file_path = os.path.join(self.path, "timelog")
        self.epoch = 0
        self.epoch_time_start = 0

    def on_train_begin(self, logs={}):
        """ Reimplementation from inherited class :class:`keras.callbacks.Callback`.
        """
        with open(self.file_path, 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(['epoch', 'time'])

    def on_epoch_begin(self, epoch, logs={}):
        """ Reimplementation from inherited class :class:`keras.callbacks.Callback`.
        """
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        """ Reimplementation from inherited class :class:`keras.callbacks.Callback`.
        """
        elapsed_time = time.time() - self.epoch_time_start

        with open(self.file_path, 'a', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow([self.epoch, elapsed_time])

        self.epoch += 1


class ValidationHistory(keras.callbacks.Callback):
    """ Custom callback object to run evaluation after epoch on arbitrary validation generator

    :param str path: A path where to store the log.
    :param object validation_set: The validation set used to evaluate the model.
    """
    def __init__(self, path, validation_set):
        super(ValidationHistory, self).__init__()
        self.path = path
        self.validation_set = validation_set
        self.file_path = os.path.join(self.path, "vallog")
        self.epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        """ Reimplementation from inherited class :class:`keras.callbacks.Callback`.
        """
        if self.epoch == 0:
            with open(self.file_path, 'w', newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerow(['epoch'] + self.model.metrics_names)

        validation = self.model.evaluate(self.validation_set)

        with open(self.file_path, 'a', newline='') as myfile:
            wr = csv.writer(myfile)
            if len(self.model.metrics_names) > 1:
                wr.writerow([self.epoch] + validation)
            else:
                wr.writerow([self.epoch, validation])

        self.epoch += 1


class ProgLogger(keras.callbacks.Callback):
    """
    A copy of the implementation in :mod:`nn.suture_detection`

    :param progress_callback: PyqtSignal which is used to emit a single `1` every time inference has finished processing
        a batch.
    :type progress_callback: PyQt5.QtCore.pyqtBoundSignal.pyqtBoundSignal
    :param cancel_callback: Callback used for triggering interupt of the inference process.
    :type cancel_callback: Any
    """
    def __init__(self, progress_callback, cancel_callback):
        super(ProgLogger, self).__init__()
        self.prog_clbk = progress_callback
        self.cancel_clbk = False  #: TODO: Not implemented yet!

    def on_predict_batch_end(self, batch, logs=None):
        """
        A copy of the implementation in :mod:`nn.suture_detection`

        :param batch: Index of the finished batch.
        :type batch: int
        :param logs: Dictionary with metric results for this batch.
        :type logs: dict
        """
        if self.cancel_clbk:
            self.model.stop_training = True
        elif self.prog_clbk:
            self.prog_clbk.emit(1)
        else:
            pass
