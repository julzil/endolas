from tensorflow import keras
import csv
import time
import os


class TimeHistory(keras.callbacks.Callback):
    """ Custom callback object to store time spend for epoch in .csv file
    """
    def __init__(self, path):
        super(TimeHistory, self).__init__()
        self.path = path
        self.file_path = os.path.join(self.path, "timelog")
        self.epoch = 0
        self.epoch_time_start = 0

    def on_train_begin(self, logs={}):
        with open(self.file_path, 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(['epoch', 'time'])

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        elapsed_time = time.time() - self.epoch_time_start

        with open(self.file_path, 'a', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow([self.epoch, elapsed_time])

        self.epoch += 1


class ValidationHistory(keras.callbacks.Callback):
    """ Custom callback object to run evaluation after epoch on arbitrary validation generator
    """
    def __init__(self, path, validation_set):
        super(ValidationHistory, self).__init__()
        self.path = path
        self.validation_set = validation_set
        self.file_path = os.path.join(self.path, "vallog")
        self.epoch = 0

    def on_epoch_end(self, epoch, logs={}):
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
