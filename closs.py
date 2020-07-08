from tensorflow import keras


# ----------------------------------------------------------------------------------------------------------------------
class EuclideanLoss(keras.losses.Loss):
    def __init__(self, batch_size=4, grid_width=5, grid_height=5, loss_type='msed'):
        """ Object used to define a custom loss for predicting a displacement field.

            Parameters
            ----------
            batch_size : int, optional
                The batch size which is used for training and evaluation.

            grid_width : int, optional
                The width of the laser grid.

            grid_height : int, optional
                The height of the laser grid

            loss_type : str, optional
                The type that can either be:
                    - 'msed' Mean Squared Euclidean Distance
                    - 'maed' Mean Absolute Euclidean Distance
                    - 'max' Maximum Euclidean Distance
                    - 'min' Minimum Euclidean Distance
        """
        if loss_type not in ['msed', 'maed', 'max', 'min']:
            raise AssertionError('Loss type "{}" not known, valid loss types are "maed", "msed, "max" and "min"'.format(loss_type))

        super().__init__(name=loss_type)
        self._batch_size = batch_size
        self._grid_width = grid_width
        self._grid_height = grid_height
        self._loss_type = loss_type

    def call(self, labels, prediction):
        """ Compute the euclidean distance loss.

        Parameters
        ----------
        labels : Tensor
            The labels forwarded by the network shape = (batch, point, x-y, mov-fix)

        prediction : Tensor
            The prediction forwarded by the network

        Returns
        -------
        float
            The loss value
        """
        loss = 0.0

        for batch_index in range(0, self._batch_size):
            ux = prediction[batch_index, :, :, 0]
            uy = prediction[batch_index, :, :, 1]

            x_mov = labels[batch_index, :, 0, 0]
            y_mov = labels[batch_index, :, 1, 0]
            x_mov_int = keras.backend.cast(x_mov, "int32")
            y_mov_int = keras.backend.cast(y_mov, "int32")

            x_fix = labels[batch_index, :, 0, 1]
            y_fix = labels[batch_index, :, 1, 1]

            ux_mov = self._get_displacement(ux, x_mov_int, y_mov_int)
            uy_mov = self._get_displacement(uy, x_mov_int, y_mov_int)

            x_squared = keras.backend.square(x_mov + ux_mov - x_fix)
            y_squared = keras.backend.square(y_mov + uy_mov - y_fix)

            sum_of_squares = x_squared + y_squared
            euclidean_distance = keras.backend.sqrt(sum_of_squares)

            if self._loss_type == 'maed':
                loss += keras.backend.mean(euclidean_distance)

            elif self._loss_type == 'msed':
                loss += keras.backend.mean(sum_of_squares)

            elif self._loss_type == 'max':
                loss += keras.backend.max(euclidean_distance)

            elif self._loss_type == 'min':
                loss += keras.backend.min(euclidean_distance)

            else:
                loss += 0.0

        loss = loss / self._batch_size

        return loss

    def _get_displacement(self, u, x, y):
        """ Use the keras backend functionality to compute the displacement in a vectorized way.

        Parameters
        ----------
        u : Tensor (width, height)
            Predicted displacement field

        x : Tensor (n_keypoints)
            x-coordinate of key point position

        y : Tensor (n_keypoints)
            y-coordinate of key point position

        Returns
        -------
        Tensor (n_keypoints)
            The displacement of each keypoint

        """
        length = self._grid_width * self._grid_height
        indices = [val * length + val for val in range(0, length)]

        u = keras.backend.gather(u, y)
        u = keras.backend.transpose(u)
        u = keras.backend.gather(u, x)
        u = keras.backend.flatten(u)
        u = keras.backend.gather(u, indices)

        return u