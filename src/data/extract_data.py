from scipy.io import loadmat
import numpy as np


def read_BCIIII_p300_mat(file_path):
    data = loadmat(file_path)
    del data['__globals__'], data['__header__'], data['__version__']
    return data


def find_timepoints_1D(single_stimulus_code):
    """
    Find the indexes where the value of single_stimulus_code turn from zero to non_zero
    single_stimulus_code : 1-D array

    >>> find_timepoints_1D([5,5,0,0,4,4,4,0,0,1,0,2,0])
    array([ 0,  4,  9, 11])
    >>> find_timepoints_1D([0,0,1,2,3,0,1,0,0])
    array([2, 6])
    >>> find_timepoints_1D([0,0,1,2,0,1])
    array([2, 5])
    >>> find_timepoints_1D([5,0,0,1,2,5])
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


def extract_data(file_path, seconds_to_slice):
    data = read_BCIIII_p300_mat(file_path)
    # Training:Signal,StimulusCode,StimulusType,TargetChar,Flashing
    # Test:Signal,StimulusCode,Flashing
    num_characters = data['Signal'].shape[0]
    num_timesteps = data['Signal'].shape[1]
    num_electrodes = data['Signal'].shape[2]
    num_intens = 12
    num_times = 15
    Fs = 240  # Sampling frequency 240Hz
    points_per_intens = int(Fs * seconds_to_slice)

    # The shape of return I want
    signal = np.zeros([num_characters, num_times, num_intens, num_electrodes, points_per_intens])
    code = np.zeros([num_characters, num_times, num_intens])
    if 'StimulusType' in data.keys():  # if true then it's training set else test set
        label = np.zeros([num_characters, num_times, num_intens])

    for character in range(num_characters):
        # All electrodes start at the same time so pick 0 is fine
        timepoints = find_timepoints_1D(data['StimulusCode'][character, :])  # (12*15,)
        timepoints = timepoints.reshape([-1, 12])  # (15,12)
        for time in range(num_times):
            for intens in range(num_intens):
                start = timepoints[time, intens]
                end = start + points_per_intens
                sliced_signal = data['Signal'][character, start:end, :]  # (end-start,64)
                sliced_signal = sliced_signal.transpose([1, 0])  # (64,end-start)
                signal[character, time, intens, :, :] = sliced_signal
                code[character, time, intens] = data['StimulusCode'][character, start]
                if 'StimulusType' in data.keys():
                    label[character, time, intens] = data['StimulusType'][character, start]
    if 'StimulusType' in data.keys():
        return {'signal': signal, 'code': code, 'label': label, 'targetchar': data['TargetChar']}
    else:
        return {'signal': signal, 'code': code}

def read_true_letters(path):
    file = open(path)
    answer_string = file.readline()
    file.close()
    return answer_string
