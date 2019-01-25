import numpy as np

from src.data.extract_data import extract_data
from src.data import preprocessing

PARADIGM = np.array([['A','B','C','D','E','F'],['G','H','I','J','K','L'],['M','N','O','P','Q','R'],
                     ['S','T','U','V','W','X'],['Y','Z','1','2','3','4'],['5','6','7','8','9','_']])


def read_preprocess_data(path, seconds_to_slice=0.65, subsample_interval=2,
                        lowcut=0.1, highcut=20, fs=240, order=5, std_axis=-1):
    """ Read data (.mat) and do the necessary preprocessing steps.

    Extract the data in .mat file into a handy structure(dict).
    Then, apply subsampling and filtering to the signal.

    Args:
        path (str): The path of the .mat file.
        seconds_to_slice (int): The number of seconds you want to slice as your data point after an intensify happens.
        subsample_interval (int): 2 means subsample by 2(halved)
        lowcut (float): The lower threshold of the filter.
        highcut (float): The upper threshold of the filter.
        fs (int): The original Fs. Change due to subsampling is handled inside the function
        order (int): The order of the filter.
        std_axis (int): along which axis do the standardization

    Return:
        dict: 
            if is_training:
                {
                    'signal': [num_characters, num_times, num_intens, num_electrodes, points_per_intens]
                    'code': [num_characters, num_times, num_intens], meaning as 'StimulusCode'
                    'label': [num_characters, num_times, num_intens], meaning as 'StimulusType'
                    'targetchar': [num_characters, num_times], meaning as 'TargetChar'
                }
            else:
                {
                    'signal'
                    'code'
                }
    """
    data = extract_data(path=path, seconds_to_slice=seconds_to_slice)
    signal = data['signal']
    subsampled = preprocessing.subsample(signal, subsample_interval=subsample_interval)
    fs = int(fs / subsample_interval)
    bandpassed = preprocessing.butter_bandpass_filter(subsampled, lowcut=lowcut, highcut=highcut,
                                                      fs=fs, order=order, axis=-1)

    # reshaped = bandpassed.reshape(reshape)
    # if expand == True:
    #     reshaped = np.expand_dims(reshaped, axis=-1)
    data['signal'] = preprocessing.standardize_along(bandpassed, axis = std_axis)
    return data


def trial_per_point(signal, expand_dims_axis=-1):
    signal = signal.reshape([-1, signal.shape[-2], signal.shape[-1]])
    if expand_dims_axis:
        signal = np.expand_dims(signal, axis=expand_dims_axis)
    return signal


def letter_lookup(data, paradigm):
    """
    data values are all 0-5,return[...,0] means which of 1-6 is selected
                                return[...,1] means which of 7-12 is selected
    :param data: [number_samples,2]
    :param paradigm: paradigm matrix ndarray
    :return: [number_samples,] ndarray
    """

    def __look_up(arr, paradigm=paradigm):
        """ arr: [2,] """
        # arr[0]  0~5->1~6  arr[1] 0~5->7~12
        return paradigm[arr[1], arr[0]]

    return np.apply_along_axis(__look_up, axis=1, arr=data)

def prob_to_rowcols(prob):
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

def code_reorder(probs, code):
    """
    after your probs has been reshaped to(85,15,12),the correspondent order of stimuli
    is recorded in data['code'],and for utility,I want to reorder them into [stimuli_1,stimuli_2,...stimuli_12]
    :param probs: [85,15,12]
    :param code: [85,15,12]
    :return: [85,15,12]
    """
    sort = np.zeros(probs.shape)
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            # sort[i, j, :] = probs[i, j, (code[i, j] - 1).astype(int).tolist()]  # This is wrong !
            sort[i, j, :] = probs[i, j, np.argsort(code[i,j])]
    return sort

def aggregate_prob_across_trials(sort,num_aggregate):
    to_aggregate = np.delete(sort, obj=np.s_[num_aggregate:], axis=1)  # (num_letters,num_aggregate,12)
    # then sum up
    aggregated = to_aggregate.sum(axis=1)   # (num_letters,12)  # aggregated probabilities
    return aggregated

def test_pipeline(signal,code, model, num_aggregate, paradigm):
    """
    num_aggregate: int 1-15 or "all"
    """
    if num_aggregate != "all":
        aggregated = signalcode_to_aggregate(signal, code, model, num_aggregate)
        return aggregated_to_letters(aggregated,paradigm)
    # (num_letters,12) -> (num_letters,2)
    # selected = prob_to_rowcols(aggregated)
    # letters = letter_lookup(selected,paradigm)
    # return letters
#########################################################################################################
def signalcode_to_aggregate(signal,code,model,num_aggregate):
    predictions = model.predict(signal)  # [num_samples,2]
    dropped = np.delete(predictions, 0, -1).squeeze()  # [num_samples,]     <- shit once (predictions,-1,-1)
    reshaped = dropped.reshape([-1, 15, 12])  # may be different
    sort = code_reorder(reshaped, code)  # [85,15,12]
    aggregated = aggregate_prob_across_trials(sort, num_aggregate)  # aggregate the probabilities
    return aggregated

def aggregated_to_letters(aggregated,paradigm):
    # (num_letters,12) -> (num_letters,2)
    selected = prob_to_rowcols(aggregated)
    letters = letter_lookup(selected, paradigm)
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

def testing_pipeline(signal,code, model, num_aggregate, paradigm,answer_string):
    """num_aggregate: int 1-15 or "all" """
    if num_aggregate != "all":        # int
        aggregated = signalcode_to_aggregate(signal, code, model, num_aggregate)
        letters = aggregated_to_letters(aggregated, paradigm)
        return [accuracy(letters,answer_string)]
    else:                           # "all"
        accuracies = []
        for i in range(15):
            aggregated = signalcode_to_aggregate(signal, code, model, i+1)
            letters = aggregated_to_letters(aggregated, paradigm)
            accuracies.append(accuracy(letters,answer_string))
        return accuracies
