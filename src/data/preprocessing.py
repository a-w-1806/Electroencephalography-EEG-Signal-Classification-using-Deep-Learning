from scipy import signal
import numpy as np

def transform(data, fs, num_rowscols, num_repeat, seconds_to_slice):
    """ Given data imported from .mat, return the data in a format which is easy to use.
    
    Args:
        data (dict): The dictionary of importing .mat file.
        fs (float): The sampling frequency.
        num_rowscols (int): The sum of the numbers of rows and columns.
        num_repeat (int): The number of times each character repeats.
        seconds_to_slice (int): The number of seconds you want to slice as your data point after an intensify happens.

    Return:
        dict: 
            if is_training:
                {
                    'signal': [num_characters, num_times, num_intens, num_electrodes, points_per_intens]
                    'code': [num_characters, num_times, num_intens], meaning as 'StimulusCode'
                    'label': [num_characters, num_times, num_intens], meaning as 'StimulusType'
                    'targetchar': [num_characters, ], meaning as 'TargetChar'. But we won't be using this to do training.
                }
            else:
                {
                    'signal'
                    'code'
                }
    """

    # data = _read_BCIIII_p300_mat(path)

    # if true then it's training set else test set
    if 'StimulusType' in data.keys():
        is_training = True
    else:
        is_training = False

    # Training: Signal, StimulusCode, StimulusType, TargetChar, Flashing
    # Test: Signal, StimulusCode, Flashing
    num_characters = data['Signal'].shape[0]
    num_electrodes = data['Signal'].shape[2]
    points_per_intens = int(fs * seconds_to_slice)

    # The shape of return I want
    signal = np.zeros([num_characters, num_repeat, num_rowscols, num_electrodes, points_per_intens])
    code = np.zeros([num_characters, num_repeat, num_rowscols])
    if is_training:  # if true then it's training set else test set
        label = np.zeros([num_characters, num_repeat, num_rowscols])

    for character in range(num_characters):
        # All electrodes start at the same time so pick 0 is fine
        timepoints = _find_timepoints_1D(data['StimulusCode'][character, :])  # (12*15,)
        timepoints = timepoints.reshape([-1, num_rowscols])  # (15,12)
        for time in range(num_repeat):
            for intens in range(num_rowscols):
                start = timepoints[time, intens]
                end = start + points_per_intens
                sliced_signal = data['Signal'][character, start:end, :]  # (end-start,64)
                sliced_signal = sliced_signal.transpose([1, 0])  # (64,end-start)
                signal[character, time, intens, :, :] = sliced_signal
                code[character, time, intens] = data['StimulusCode'][character, start]
                if is_training:
                    label[character, time, intens] = data['StimulusType'][character, start]
    if is_training:
        return {'signal': signal, 'code': code, 'label': label, 'targetchar': np.array(list(data['TargetChar'][0]))}
    else:
        return {'signal': signal, 'code': code}

def _find_timepoints_1D(single_stimulus_code):
    """
    Find the indexes where the value of single_stimulus_code turn from zero to non_zero
    single_stimulus_code : 1-D array

    >>> _find_timepoints_1D([5,5,0,0,4,4,4,0,0,1,0,2,0])
    array([ 0,  4,  9, 11])
    >>> _find_timepoints_1D([0,0,1,2,3,0,1,0,0])
    array([2, 6])
    >>> _find_timepoints_1D([0,0,1,2,0,1])
    array([2, 5])
    >>> _find_timepoints_1D([5,0,0,1,2,5])
    array([0, 3])
    """
    flag = True  # whether have seen 0 so far
    timepoints = []
    for index, timepoint in enumerate(single_stimulus_code):
        if timepoint != 0 and flag:
            timepoints.append(index)
            flag = False
        if timepoint == 0 and not flag:
            flag = True
    return np.array(timepoints)

def subsample(data_array, subsample_interval):
    """  Subsample every points_per_sample points """
    return data_array[..., 0::subsample_interval]

def __butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=-1):
    b, a = __butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data, axis=axis)
    return y


def standardize_along(data, axis):
    mean = data.mean(axis = axis, keepdims = True)
    std = data.std(axis = axis,keepdims = True)
    return (data-mean)/std