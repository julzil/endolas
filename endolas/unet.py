import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Input, Conv2D, MaxPooling2D, Activation, UpSampling2D, Concatenate
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import imagenet_utils


def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    :param ndarray x: a 4D numpy array consists of RGB values within [0, 255].
    :return: Preprocessed array
    :rtype: ndarray
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)


def UNet(filters=64, layers=4, activation='sigmoid', classes=1, input_shape=None,
         kernel_regularizer=None, gap_filling=False):
    """
    Building a U-Net [#ronneberger]_. Implementation from [#ankigit]_ modified.

    .. [#ronneberger] 'Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for
                      biomedical image segmentation. In International Conference on Medical image computing and
                      computer-assisted intervention (pp. 234-241). Springer, Cham.'
    .. [#ankigit] `Github <https://github.com/anki-xyz/bagls/blob/master>`_ Repository from Anki

    :param int filters: The number of filters in the first layer.
                        The subsequent layers have multiples of this filter number. Default is 64.
    :param int layers: The number of encoding and decoding layers. Default is 4.
    :param str activation: The activation function in the last layer. Default is sigmoid.
    :param int classes: The number of classes in the last layer. Default is 1.
    :param tuple input_shape: The input shape of the data. We train the network to have arbitrary
                              input shapes, default is None. Otherwise, the tuple has to follow
                              the following criterion: (X, Y, channels)
    :param str kernel_regularizer: The kernel regularizer used for all Conv2D layers.
    :param bool gap_filling: Whether gap filling technique should be used or not.
    :return: A Keras Model containing the U-Net structure.
    :rtype: keras model
    """
    if input_shape is None:
        input_shape = (None, None, 1)

    model_input = Input(shape=input_shape)

    to_concat = []

    x = model_input

    # Encoding
    for block in range(layers):
        x = _convblock(x, filters * (2 ** block), kernel_regularizer)
        x = _convblock(x, filters * (2 ** block), kernel_regularizer)

        if gap_filling:
            gap_layer = _convblock(x, filters * (2 ** block), kernel_regularizer)
            additional_gap_layers = 4 * (layers - block - 1)
            for index in range(additional_gap_layers):
                gap_layer = _convblock(gap_layer, filters * (2 ** block), kernel_regularizer)
            to_concat.append(gap_layer)
            
        else:
            to_concat.append(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = _convblock(x, filters * (2 ** (block + 1)), kernel_regularizer)

    # Decoding
    for block, filter_factor in enumerate(np.arange(layers)[::-1]):
        x = _convblock(x, filters * (2 ** filter_factor), kernel_regularizer)

        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate()([x, to_concat[::-1][block]])

        x = _convblock(x, filters * (2 ** filter_factor), kernel_regularizer)

    x = _convblock(x, filters * (2 ** filter_factor), kernel_regularizer)

    # Final output, 1x1 convolution
    segmentation_output = Conv2D(classes,
                                 (1, 1),
                                 use_bias=False,
                                 padding="same",
                                 activation=activation,
                                 strides=1,
                                 kernel_initializer='glorot_uniform',
                                 kernel_regularizer=kernel_regularizer)(x)

    return Model(inputs=model_input, outputs=segmentation_output)


def _convblock(x, filters, kernel_regularizer, batch_norm=True):
    """ Convolutional block with batch norm and activation.
    """
    x = Conv2D(filters,
               (3, 3),
               use_bias=False,
               padding="same",
               strides=1,
               kernel_initializer='he_uniform', # glorot_uniform
               kernel_regularizer=kernel_regularizer)(x)

    if batch_norm:
        x = BatchNormalization()(x)

    x = Activation("relu")(x)

    return x
