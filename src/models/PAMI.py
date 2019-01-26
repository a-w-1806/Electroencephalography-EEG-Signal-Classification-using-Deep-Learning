import tensorflow as tf
from keras import layers, models
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import numpy as np


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

def CNN_P300_PAMI(input_shape):
    """ A Deep Learning Model inspired by a paper on PAMI.

    Reference: 
        H. Cecotti and A. Graser, "Convolutional Neural Networks for P300 Detection with Application to Brain-Computer Interfaces," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 3, pp. 433-445, March 2011.
        doi: 10.1109/TPAMI.2010.125

    Return:
        An uncompiled (Keras jargon) model.
    """

    # Let's define the custom activation function used in this layer first
    def sigmoid_pami_p300(x):
        return 1.7159 * K.tanh(2 * x / 3)
    get_custom_objects().update({'custom_activation': layers.Activation(sigmoid_pami_p300)})

    # Then we build the model
    num_electrodes, points_to_slice = input_shape[0:2]
    num_first_filter = 10
    num_second_filter = 5 * num_first_filter

    input_layer = layers.Input(input_shape)
    # Combining all the electrodes.
    conv_1 = layers.convolutional.Conv2D(filters=num_first_filter, kernel_size=[num_electrodes, 1], 
                                        strides=[1, 1], activation='linear')(input_layer)
    conv_1_acti = layers.Activation(sigmoid_pami_p300)(conv_1)

    # It's slightly different in the paper though..(filter have same parameter along the same axis)
    # But I decide to use this more modern choice.
    # Have the senmantic of averaging across time.
    conv_2 = layers.convolutional.Conv2D(filters=num_second_filter, kernel_size=[1, 13], 
                                        strides=[1, 13], activation='linear')(conv_1_acti)
    conv_2_acti = layers.Activation(sigmoid_pami_p300)(conv_2)
    flatten = layers.Flatten()(conv_2_acti)
    fc = layers.Dense(100, activation='sigmoid')(flatten)
    output_layer = layers.Dense(2, activation='sigmoid')(fc)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model

def to_chars(data, trained_model):
    """ Given the input data of this model, return the characters of each trial as final result.

    Args:
        data (dict):
            {
                'signal': [-1, num_electrodes, num_samples, 1]
                'code': [num_chars, num_repeats, num_rowcols]
            } 
    Return:
        np.array of chars: The classification result of this result.
    """
    num_electrodes = data['signal'].shape[1]
    num_chars, num_repeats, num_rowcols = data['code'].shape

    predictions = trained_model.predict(data['signal']) # [-1, 2]
    #TODO Sigmoid should output only 1.
    predictions = np.delete(predictions, 0, -1).squeeze()

    # [num_chars, num_repeats, num_rowcols]
    # Each entry is the probability of a positive sample (when intensified row/character does contain the desired character).
    predictions = predictions.reshape([-1, num_repeats, num_rowcols]) 

    predictions = _code_reorder(predictions, data['code'])


# def _to_code(predictions, code):
#     result = np.zeros(predictions.shape)
#     for character in range(predictions.shape[0]):
#         for repeat in range(predictions.shape[1]):
#             result[character][repeat] = 
def _code_reorder(probs, code):
    """ Utility function to reorder the last index of probs according to the order of occurrence of the code.

    The correspondent order of code is recorded in data['code'], and for utility, I want to reorder them into [code_1, code_2, ...code_12]

    Args:
        probs (np.array): [num_chars, num_repeats, num_rowcols]. Each entry is the probability of a positive sample (when intensified row/character does contain the desired character).
        code (np.array): [num_chars, num_repeats, num_rowcols]. The occurrence order of each code in each trial.
    
    Returns:
        np.array: [num_chars, num_repeats, num_rowcols]. 
    """
    sort = np.zeros(probs.shape)
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            # sort[i, j, :] = probs[i, j, (code[i, j] - 1).astype(int).tolist()]  # This is wrong !
            sort[i, j, :] = probs[i, j, np.argsort(code[i,j])]
    return sort