from scipy.io import loadmat
import numpy as np

def extract_data(file_path, seconds_to_slice):
    """ Given a file_path, return the data in a format which is easy to use.
    
    Args:
        file_path (string): The path of the given .mat file.
        seconds_to_slice (int): The number of seconds you want to slice as your data point after an intensify happens.

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

    # if true then it's training set else test set
    if 'StimulusType' in data.keys():
        is_training = True
    else:
        is_training = False

    data = _read_BCIIII_p300_mat(file_path)
    # Training:Signal,StimulusCode,StimulusType,TargetChar,Flashing
    # Test:Signal,StimulusCode,Flashing
    num_characters = data['Signal'].shape[0]
    num_timesteps = data['Signal'].shape[1]
    num_electrodes = data['Signal'].shape[2]
    num_intens = 12 # 6 rows + 6 cols
    num_times = 15  # Each character is runned for 15 times, so 12 * 15 = 180 intensifies
    Fs = 240  # Sampling frequency 240Hz
    points_per_intens = int(Fs * seconds_to_slice)

    # The shape of return I want
    signal = np.zeros([num_characters, num_times, num_intens, num_electrodes, points_per_intens])
    code = np.zeros([num_characters, num_times, num_intens])
    if is_training:  # if true then it's training set else test set
        label = np.zeros([num_characters, num_times, num_intens])

    for character in range(num_characters):
        # All electrodes start at the same time so pick 0 is fine
        timepoints = _find_timepoints_1D(data['StimulusCode'][character, :])  # (12*15,)
        timepoints = timepoints.reshape([-1, num_intens])  # (15,12)
        for time in range(num_times):
            for intens in range(num_intens):
                start = timepoints[time, intens]
                end = start + points_per_intens
                sliced_signal = data['Signal'][character, start:end, :]  # (end-start,64)
                sliced_signal = sliced_signal.transpose([1, 0])  # (64,end-start)
                signal[character, time, intens, :, :] = sliced_signal
                code[character, time, intens] = data['StimulusCode'][character, start]
                if is_training:
                    label[character, time, intens] = data['StimulusType'][character, start]
    if is_training:
        return {'signal': signal, 'code': code, 'label': label, 'targetchar': data['TargetChar']}
    else:
        return {'signal': signal, 'code': code}

def _read_BCIIII_p300_mat(file_path):
    """ Read .mat file into a dict """
    data = loadmat(file_path)
    del data['__globals__'], data['__header__'], data['__version__']
    return data

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

def read_true_letters(path):
    file = open(path)
    answer_string = file.readline()
    file.close()
    return answer_string
