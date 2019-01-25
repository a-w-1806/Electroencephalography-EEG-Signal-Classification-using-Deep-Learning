import tensorflow as tf
from keras import layers, models
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


# In the original paper they use a custom activation function for
# the first two layers




# model.add(Activation(custom_activation))

def CNN_1_P300_PAMI_BCIIII(Ns=10, seconds_to_slice=0.65):
    """ Reference: Convolutional Neural Networks for P300 Detection with Application to Brain-Computer Interfaces

    return an uncompiled model

    Ns: Number of filters in the first conv layer
    """

    # Let's define the custom activation function used in this layer first
    def sigmoid_pami_p300(x):
        return 1.7159 * K.tanh(2 * x / 3)

    get_custom_objects().update({'custom_activation': layers.Activation(sigmoid_pami_p300)})

    # Then we build the model
    num_electrodes = 64
    Fs = 120  # Original Fs is 240Hz but in this paper they have subsampled
    points_to_slice = int(Fs * seconds_to_slice)
    input_layer = layers.Input([num_electrodes, points_to_slice, 1])
    conv_1 = layers.convolutional.Conv2D(filters=Ns, kernel_size=[num_electrodes, 1], strides=[1, 1],
                                         activation='linear')(input_layer)

    conv_1_acti = layers.Activation(sigmoid_pami_p300)(conv_1)
    # It's slightly different in the paper though..(filter have same parameter along the same axis)but I decide to use this more modern choice
    conv_2 = layers.convolutional.Conv2D(filters=5 * Ns, kernel_size=[1, 13], strides=[1, 13],
                                         activation='linear')(conv_1_acti)
    conv_2_acti = layers.Activation(sigmoid_pami_p300)(conv_2)
    flatten = layers.Flatten()(conv_2_acti)
    fc = layers.Dense(100, activation='sigmoid')(flatten)
    output_layer = layers.Dense(2, activation='sigmoid')(fc)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model
