import tensorflow as tf
from keras import layers, models
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import numpy as np

from src.data.paradigm import PARADIGM


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

def testing_pipeline(data, trained_model, num_aggregate, answer_string):
    """ A utility function to output the final performance of the model on the data.
    
    """
    if num_aggregate != "all":        # int
        aggregated = _signalcode_to_aggregate(data, trained_model, num_aggregate)
        letters = _aggregated_to_letters(aggregated)
        return accuracy(letters,answer_string)
    else:                           # "all"
        accuracies = []
        for i in range(15):
            accuracies.append(testing_pipeline(data, trained_model, i+1, answer_string))
        return accuracies

def _letter_lookup(data):
    """
    data values are all 0-5,return[...,0] means which of 1-6 is selected
                                return[...,1] means which of 7-12 is selected
    :param data: [number_samples,2]
    :param paradigm: paradigm matrix ndarray
    :return: [number_samples,] ndarray
    """

    def __look_up(arr, paradigm=PARADIGM):
        """ arr: [2,] """
        # arr[0]  0~5->1~6  arr[1] 0~5->7~12
        return paradigm[arr[1], arr[0]]

    return np.apply_along_axis(__look_up, axis=1, arr=data)

def _prob_to_rowcols(prob):
    """
    returned values are all 0-5,return[...,0] means which of 1-6 is selected
                                return[...,1] means which of 7-12 is selected
    :param prob: (num_letters,12)   accumulated probabilities
    :return: (num_letters,2) row/column  0-5->1-6 / 0-5->7-12
    """
    ascend_indices = np.argsort(prob, axis=-1)  # (num_letters,12)     # ascending
    selected = np.full((prob.shape[0], 2), np.inf)
    for i in range(selected.shape[0]):
        for j in reversed(range(ascend_indices.shape[1])):
            if ascend_indices[i][j] >= 6 and selected[i][1] == np.inf:
                selected[i][1] = ascend_indices[i][j] - 6
            elif ascend_indices[i][j] <= 5 and selected[i][0] == np.inf:
                selected[i][0] = ascend_indices[i][j]
    return selected.astype(int)

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
            sort[i, j, :] = probs[i, j, np.argsort(code[i,j])]
    return sort

def _aggregate_prob_across_trials(sort,num_aggregate):
    to_aggregate = np.delete(sort, obj=np.s_[num_aggregate:], axis=1)  # (num_letters,num_aggregate,12)
    # then sum up
    aggregated = to_aggregate.sum(axis=1)   # (num_letters,12)  # aggregated probabilities
    return aggregated

def _signalcode_to_aggregate(data,model,num_aggregate):
    predictions = model.predict(data['signal'])  # [num_samples,2]
    dropped = np.delete(predictions, 0, -1).squeeze()  # [num_samples,]     <- shit once (predictions,-1,-1)
    reshaped = dropped.reshape([-1, 15, 12])  # may be different
    sort = _code_reorder(reshaped, data['code'])  # [85,15,12]
    aggregated = _aggregate_prob_across_trials(sort, num_aggregate)  # aggregate the probabilities
    return aggregated

def _aggregated_to_letters(aggregated):
    # (num_letters,12) -> (num_letters,2)
    selected = _prob_to_rowcols(aggregated)
    letters = _letter_lookup(selected)
    return letters

def accuracy(letters, target_string):
    """

    :param letters: ndarray (num_letters,)
    :param target_string string
    :return: float
    """
    count = 0
    assert len(letters) == len(target_string)
    for i in range(len(target_string)):
        if letters[i] == target_string[i]:
            count+=1
    return count/len(target_string)
    
    