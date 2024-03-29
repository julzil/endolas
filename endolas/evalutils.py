import numpy as np
import json
import pandas as pd
import os
import tensorflow as tf

from glob import glob
from .closs import EuclideanLoss
from .lastengen import LASTENSequence


def eval_pred(path_fixed, path_validation, path_test, store_path, width, height, grid_width,
              grid_height, pre_une, weights='/weights.100.hdf5'):
    """ Predict a validation and test set and store in file.

    :param str path_fixed: Path to the fixed image.
    :param str path_validation: Path to the validation data set.
    :param str path_test: Path to the test data set.
    :param str store_path: Path to where the results should be stored.
    :param int width: Width for resizing in the generator sequence.
    :param int height: Height for resizing in the generator sequence.
    :param int grid_width: The width of the laser grid, for example 18 points.
    :param int grid_height: The height of the laser grid, for example 18 points.
    :param function pre_une: Function needed by the generators for preprocessing data.
    :param object weights: A path to the keras network including weights.
    """
    test_gen = LASTENSequence(path_test,
                              path_fixed,
                              batch_size=1,
                              width=width,
                              height=height,
                              grid_width=grid_width,
                              grid_height=grid_height,
                              preprocess_input=pre_une,
                              shuffle=True,
                              label="keypoints",
                              channel="moving+fixed")

    validation_gen = LASTENSequence(path_validation,
                                    path_fixed,
                                    batch_size=1,
                                    width=width,
                                    height=height,
                                    grid_width=grid_width,
                                    grid_height=grid_height,
                                    preprocess_input=pre_une,
                                    shuffle=False,
                                    label="keypoints",
                                    channel="moving+fixed")

    eu_loss = EuclideanLoss(batch_size=1, grid_width=grid_width, grid_height=grid_height, loss_type='med')

    #dependencies = {'loss': eu_loss}
    model = tf.keras.models.load_model(store_path + weights, compile=False)
    model.compile(loss=eu_loss)

    grid_points = grid_width * grid_height

    warp_val = dict()
    maed_val_array = np.zeros(len(validation_gen))
    for index, val in enumerate(validation_gen):
        image_id = validation_gen._image_ids[index]

        if index % 100 == 0:
            print(index)
        X, y = val
        maed = model.evaluate(X ,y ,verbose=0)
        maed_val_array[index] = maed

        y_pred = model.predict(X)
        u_x = y_pred[0 ,: ,: ,0]
        u_y = y_pred[0 ,: ,: ,1]

        for inner_index in range(0 ,grid_points):
            x_pos = int(y[0, inner_index, 0, 0])
            y_pos = int(y[0, inner_index, 1, 0])

            ux_field = y_pred[0 ,: ,: ,0]
            uy_field = y_pred[0 ,: ,: ,1]

            ux = ux_field[y_pos][x_pos]
            uy = uy_field[y_pos][x_pos]

            x_pos = int(round(x_pos + ux))
            y_pos = int(round(y_pos + uy))

            warp_val[str(inner_index)] = [x_pos, y_pos]
            with open(store_path +'/val/{}_w.json'.format(image_id), 'w') as fp:
                json.dump(warp_val, fp)

    warp_test = dict()
    maed_test_array = np.zeros(len(test_gen))
    for index, val in enumerate(test_gen):
        image_id = test_gen._image_ids[index]

        if index % 100 == 0:
            print(index)
        X, y = val
        maed = model.evaluate(X, y, verbose=0)
        maed_test_array[index] = maed

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

            warp_test[str(inner_index)] = [x_pos, y_pos]
            with open(store_path + '/test/{}_w.json'.format(image_id), 'w') as fp:
                json.dump(warp_test, fp)

    maed_array = np.concatenate((maed_val_array, maed_test_array))
    val_set = ['val' for i in range(len(validation_gen))]
    test_set = ['test' for i in range(len(test_gen))]
    set_type = val_set + test_set
    image_id = validation_gen._image_ids + test_gen._image_ids

    dataset = pd.DataFrame({'MED': maed_array, 'Set': set_type, 'Image': image_id})
    dataset.to_csv(store_path + '/evaluation.csv')


def eval_nearest_neighbor(experiment_2_set_2_image_2_accuracy, store_path):
    """ Evaluate the nearest neighbor results by writing them to a pandas dataframe.

    :param dict experiment_2_set_2_image_2_accuracy: A mapping containing results of several experiments.
    :param str store_path: The path to store the pandas dataframe to.
    """
    accuracys = []
    images = []
    set_types = []
    experiments = []

    for experiment, set_2_image_2_accuracy in experiment_2_set_2_image_2_accuracy.items():
        for set_type, image_2_accuracy in set_2_image_2_accuracy.items():
            for image, accuracy in image_2_accuracy[0].items():
                accuracys.append(accuracy)
                images.append(image)
                set_types.append(set_type)
                experiments.append(experiment)

    dataset = pd.DataFrame({'Accuracy': accuracys, 'Image': images, 'Set': set_types, 'Experiment': experiments})
    dataset.to_csv(store_path)


def spatial_distribution(store_path):
    """ Evaluate the spatial distribution of warped keypoints by writing them to a pandas dataframe.

    :param str store_path: A path were warped keypoints are stored and will be written to
    """
    x_val = []
    y_val = []
    image = []
    point = []
    set_types = []

    for set_type in ['val', 'test']:
        data_path = os.path.join(store_path, set_type)
        globs = glob(data_path + os.sep + "*_w.json")
        globs = [int(path.split(os.sep)[-1].split(".")[0].split("_")[0]) for path in globs]
        image_ids = sorted(globs)

        for image_id in image_ids:
            # 0)Load data
            warp_path = data_path + os.sep + "{}_w.json".format(image_id)
            with open(warp_path) as warped_file:
                warped_json = json.load(warped_file)

            # 1) Sort out all obsolete points
            for key, value in list(warped_json.items()):
                if value[0] < 2.0 and value[1] < 2.0:
                    _ = warped_json.pop(key)

            # 2) Build arrays
            for key, value in list(warped_json.items()):
                x_val.append(value[0])
                y_val.append(value[1])
                image.append(image_id)
                point.append(key)
                set_types.append(set_type)

    # 3) Build dataframe
    dataset = pd.DataFrame({'x': x_val, 'y': y_val, 'Set': set_types, 'Image': image, 'Point': point})
    dataset.to_csv(store_path + '/evaluation_spatial.csv')


def spatial_display(store_path, path_validation, path_test):
    """ Evaluate the spatial distribution for displaying by writing them to a pandas dataframe.

    :param str store_path: Path were warped keypoints are stored.
    :param str path_validation: Path were moving keypoints from validation are stored.
    :param str path_test: Path were moving keypoints from test are stored.
    """
    x_val = []
    y_val = []
    image = []
    point = []
    set_types = []
    image_types = []

    # Warp
    for set_type in ['val', 'test']:
        data_path = os.path.join(store_path, set_type)
        globs = glob(data_path + os.sep + "*_w.json")
        globs = [int(path.split(os.sep)[-1].split(".")[0].split("_")[0]) for path in globs]
        image_ids = sorted(globs)

        for image_id in image_ids:
            # 0)Load data
            warp_path = data_path + os.sep + "{}_w.json".format(image_id)
            with open(warp_path) as warped_file:
                warped_json = json.load(warped_file)

            # 1) Sort out all obsolete points
            for key, value in list(warped_json.items()):
                if value[0] < 2.0 and value[1] < 2.0:
                    _ = warped_json.pop(key)

            # 2) Build arrays
            for key, value in list(warped_json.items()):
                x_val.append(value[0])
                y_val.append(value[1])
                image.append(image_id)
                point.append(key)
                set_types.append(set_type)
                image_types.append('warped')

    # Moving validation
    globs = glob(path_validation + os.sep + "*.json")
    globs = [int(path.split(os.sep)[-1].split(".")[0].split("_")[0]) for path in globs]
    image_ids = sorted(globs)

    for image_id in image_ids:
        # 0)Load data
        moving_path = path_validation + os.sep + "{}.json".format(image_id)
        with open(moving_path) as moving_file:
            moving_json = json.load(moving_file)

        # 1) Sort out all obsolete points
        for key, value in list(moving_json.items()):
            if value[0] < 2.0 and value[1] < 2.0:
                _ = warped_json.pop(key)

        # 2) Build arrays
        for key, value in list(moving_json.items()):
            x_val.append(value[0])
            y_val.append(value[1])
            image.append(image_id)
            point.append(key)
            set_types.append('val')
            image_types.append('moving')

    # Moving test
    globs = glob(path_test + os.sep + "*.json")
    globs = [int(path.split(os.sep)[-1].split(".")[0].split("_")[0]) for path in globs]
    image_ids = sorted(globs)

    for image_id in image_ids:
        # 0)Load data
        moving_path = path_test + os.sep + "{}.json".format(image_id)
        with open(moving_path) as moving_file:
            moving_json = json.load(moving_file)

        # 1) Sort out all obsolete points
        for key, value in list(moving_json.items()):
            if value[0] < 2.0 and value[1] < 2.0:
                _ = warped_json.pop(key)

        # 2) Build arrays
        for key, value in list(moving_json.items()):
            x_val.append(value[0])
            y_val.append(value[1])
            image.append(image_id)
            point.append(key)
            set_types.append('test')
            image_types.append('moving')

    # Build dataframe
    dataset = pd.DataFrame(
        {'x': x_val, 'y': y_val, 'Set': set_types, 'Image': image, 'Point': point, 'Type': image_types})
    dataset.to_csv(store_path + '/evaluation_display.csv')


def store_accuracy(accuracy_val, accuracy_test, store_path):
    """ Store the accuracy values in a pandas dataframe.

    :param dict accuracy_val: Accuracy values of the validation set.
    :param dict accuracy_test: Accuracy values of the test set.
    :param str store_path: Path to store the pandas dataframe.
    """
    set_2_image_2_accuracy = dict()
    set_2_image_2_accuracy['val'] = accuracy_val
    set_2_image_2_accuracy['test'] = accuracy_test

    accuracys = []
    images = []
    set_types = []

    for set_type, image_2_accuracy in set_2_image_2_accuracy.items():
        for image, accuracy in image_2_accuracy.items():
            accuracys.append(accuracy)
            images.append(image)
            set_types.append(set_type)

    dataset = pd.DataFrame({'Accuracy': accuracys, 'Image': images, 'Set': set_types})
    dataset.to_csv(store_path+os.sep+"accuracy.csv")


def eval_history(path_fixed, path_data, store_path, width, height, grid_width,
                 grid_height, pre_une, image_id):
    """ Evaluate the data for multiple frames.

    :param str path_fixed: Path to the fixed image.
    :param str path_data: The path were data is pulled from by generator.
    :param str store_path: Path to where the results should be stored.
    :param int width: Width for resizing in the generator sequence.
    :param int height: Height for resizing in the generator sequence.
    :param int grid_width: The width of the laser grid, for example 18 points.
    :param int grid_height: The height of the laser grid, for example 18 points.
    :param function pre_une: Function needed by the generators for preprocessing data.
    :param int image_id: The ID of the respective image.
    """
    gen = LASTENSequence(path_data,
                         path_fixed,
                         image_ids=[image_id],
                         batch_size=1,
                         width=width,
                         height=height,
                         grid_width=grid_width,
                         grid_height=grid_height,
                         preprocess_input=pre_une,
                         shuffle=True,
                         label="keypoints",
                         channel="moving+fixed")

    eu_loss = EuclideanLoss(batch_size=1, grid_width=grid_width, grid_height=grid_height, loss_type='maed')

    epochs = [10, 20, 30, 40, 50, 60]
    for epoch in epochs:
        tf.keras.backend.clear_session()

        model = tf.keras.models.load_model(store_path + "/weights.{}.hdf5".format(epoch), compile=False)
        model.compile(loss=eu_loss)

        grid_points = grid_width * grid_height

        warp_test = dict()
        maed_test_array = np.zeros(len(gen))
        for index, val in enumerate(gen):
            image_id = gen._image_ids[index]

            if index % 100 == 0:
                print(index)
            X, y = val
            maed = model.evaluate(X, y, verbose=0)
            maed_test_array[index] = maed

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

                warp_test[str(inner_index)] = [x_pos, y_pos]
                with open(store_path + '/history/{}_w_{}.json'.format(image_id, epoch), 'w') as fp:
                    json.dump(warp_test, fp)


def spatial_history(store_path, image_id, warped_key_2_ismis, warped_key_2_fixed_key, last_epoch):
    """ For multiple frames store results of spatial location in a pandas dataframe.

    :param str store_path: A path to were the data should be stored.
    :param int image_id: The ID of the current image.
    :param dict warped_key_2_ismis: A mapping from warped keys to a bool if the keypoint is misclassified
    :param dict warped_key_2_fixed_key: A mapping from warped_keys to fixed_keys
    :param int last_epoch: The last epoch used to evaluate the misclassification
    """
    x_val = []
    y_val = []
    point = []
    epoch_write = []
    is_mis = []
    fixed_key = []

    store_path = store_path + '/history'

    # Warp
    epochs = [10, 20, 30, 40, 50, 60]

    for epoch in epochs:
        # 0)Load data
        warp_path = store_path + os.sep + "{}_w_{}.json".format(image_id, epoch)
        with open(warp_path) as warped_file:
            warped_json = json.load(warped_file)

        # 1) Sort out all obsolete points
        for key, value in list(warped_json.items()):
            if value[0] < 2.0 and value[1] < 2.0:
                _ = warped_json.pop(key)

        # 2) Build arrays
        for key, value in list(warped_json.items()):
            x_val.append(value[0])
            y_val.append(value[1])
            point.append(key)
            epoch_write.append(epoch)
            if epoch==last_epoch:
                is_mis.append(warped_key_2_ismis[str(key)])
                fixed_key.append(warped_key_2_fixed_key[str(key)])

            else:
                is_mis.append(1)
                fixed_key.append(0)

    # Build dataframe
    dataset = pd.DataFrame(
        {'x': x_val, 'y': y_val, 'Point': point, 'Epoch': epoch_write, 'Miss': is_mis, 'Fixed': fixed_key})
    dataset.to_csv(store_path + '/evaluation_history.csv')
