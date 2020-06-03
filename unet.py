import keys

import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Input, Conv2D, MaxPooling2D, Activation, UpSampling2D, Concatenate
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import imagenet_utils


def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    Parameters
    ----------

    x:
        a 4D numpy array consists of RGB values within [0, 255].

    Returns
    -------

    Preprocessed array

    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)


def UNet(filters=64, layers=4, activation='sigmoid', classes=1, input_shape=None):
    """
    Building a U-Net [1]_. Implementation from [2]_ modified for a keypoint output.


    Parameters
    ----------

    filters : int, optional
        The number of filters in the first layer.
        The subsequent layers have multiples of this filter number.
        Default is 64.

    layers : int, optional
        The number of encoding and decoding layers. Default is 4.

    activation : str, optional
        The activation function in the last layer. Default is sigmoid.

    classes : int, optional
        The number of classes in the last layer. Default is 1.

    input_shape : tuple, optional
        The input shape of the data. We train the network to have arbitraty
        input shapes, default is None. Otherwise, the tuple has to follow
        the following criterion: (X, Y, channels)

    keypoints : int, optional
        The number of keypoints. Default is 2.

    encoder : str, optional
        The encoder to be used for the network. Possible options are 'classic', 'mobilenetv2' and 'efficientnet'.
        Ensure to make use of the correct preprocessing function.


    Returns
    -------

    Keras Model
        A Keras Model containing the U-Net structure.


    References
    ----------

    [1] Ronneberger, O., Fischer, P., & Brox, T. (2015, October).
    U-net: Convolutional networks for biomedical image segmentation.
    In International Conference on Medical image computing and
    computer-assisted intervention (pp. 234-241). Springer, Cham.

    [2] https://github.com/anki-xyz/bagls/blob/master/Utils/DataGenerator.py

    """
    if input_shape is None:
        input_shape = (None, None, 1)

    model_input = Input(shape=input_shape)

    to_concat = []

    x = model_input

    # Encoding
    for block in range(layers):
        x = _convblock(x, filters * (2 ** block))
        x = _convblock(x, filters * (2 ** block))
        to_concat.append(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = _convblock(x, filters * (2 ** (block + 1)))

    # Decoding
    for block, filter_factor in enumerate(np.arange(layers)[::-1]):
        x = _convblock(x, filters * (2 ** filter_factor))

        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate()([x, to_concat[::-1][block]])

        x = _convblock(x, filters * (2 ** filter_factor))

    x = _convblock(x, filters * (2 ** filter_factor))

    # Final output, 1x1 convolution
    segmentation_output = Conv2D(classes,
                                 (1, 1),
                                 use_bias=True,
                                 padding="same",
                                 activation=activation,
                                 strides=1,
                                 kernel_initializer='glorot_uniform',
                                 name=keys.SEGMENTATION_OUTPUT_NAME)(x)

    return Model(inputs=model_input, outputs=segmentation_output)


def _convblock(x, filters, batch_norm=True):
    """ The implementation was copied from [1]_.

    [1] https://github.com/anki-xyz/bagls/blob/master/Utils/DataGenerator.py

    """
    x = Conv2D(filters,
               (3, 3),
               use_bias=True,
               padding="same",
               strides=1,
               kernel_initializer='he_uniform')(x)  # glorot_uniform

    if batch_norm:
        x = BatchNormalization()(x)

    x = Activation("relu")(x)

    return x
