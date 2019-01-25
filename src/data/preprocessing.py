from scipy import signal


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