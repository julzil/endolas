from tensorflow import keras
import csv
import time
import os

class TimeHistory(keras.callbacks.Callback):
    """ Custom callback object to store time spend for epoch in .csv file
    """
    def __init__(self, path):
        super().__init__()
        self.path = path

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

    def on_train_end(self, logs={}):
        file_path = os.path.join(self.path, "timelog")
        with open(file_path, 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(['epoch', 'time'])
            counter=0

            for time_value in self.times:
                wr.writerow([counter, time_value])
                counter += 1
