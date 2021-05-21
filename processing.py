import scipy.signal
import numpy as np
import numba


def filter_signal(ts, rate, low_freq=None, high_freq=None, order=4):
    if low_freq:
        lb_n_freq = low_freq / (rate/2)
        b, a = scipy.signal.butter(order, lb_n_freq, 'high')
        ts = scipy.signal.filtfilt(b, a, ts)
        
    if high_freq:
        hb_n_freq = high_freq / (rate/2)
        b, a = scipy.signal.butter(order, hb_n_freq, 'low')
        ts = scipy.signal.filtfilt(b, a, ts)
     
    ts = ts.copy()
    return ts



def process_ecg(ecg, trend=100, downsample=2, low_freq=None, high_freq=40, clip=20, divide_by=50):
    processed = ecg.astype('float32')
    for i in range(2):
        s = processed[i]
        mask = s != 0
        s[:100] = np.median(s[mask][:1000])
        s[-100:] = np.median(s[mask][-1000:])
        s = interp_holes(s)

        s -= np.median(s)
        kernel = np.ones(trend) / trend
        mean = np.convolve(s, kernel, mode='same')
        mean = np.convolve(mean, kernel, mode='same')
        s -= mean    
        s = filter_signal(s, 250, low_freq=low_freq, high_freq=high_freq)
        s /= divide_by
        s = np.clip(s, -clip, +clip)
        processed[i] = s
        
    processed = processed[:2, :].astype('float16')
    processed = processed[:, ::downsample]
    return processed



@numba.njit
def interp_holes(arr):
    zeros = np.where(arr == 0)[0]
    if len(zeros) == 0:
        return arr
    
    first = zeros[0]
    last = zeros[0]
    for zero in zeros[1:]:
        if zero - last == 1:
            last = zero
        else:
            left = arr[first - 1]
            right = arr[last + 1]
            arr[first - 1: last + 2] = np.linspace(left, right, last - first + 3)
            first = zero
            last = zero
     
    left = arr[first - 1]
    right = arr[last + 1]
    arr[first - 1: last + 2] = np.linspace(left, right, last - first + 3)    
    
    return arr


def get_slice(ecg, center, window, maxval=20, threshold=10):
    start = center - window // 2
    if start < 0:
        start = 0
        end = window
    else:
        end = start + window
    
    if end > ecg.shape[1]:
        end = ecg.shape[1]
        start = end - window
        
    assert start >=0 and end <= ecg.shape[1], f'Start: {start}, End: {end}, Shape: {ecg.shape[1]}'
        
    ecg_slice = ecg[:, start: end]
    if ((ecg_slice == -maxval) | (ecg_slice == maxval)).sum() < threshold:
        ecg_slice -= np.median(ecg_slice, axis=1)[:, None]
        std = ecg_slice.std(axis=1)[:, None]
        std[std == 0] = 1e-6
        ecg_slice /= std
        ecg_slice = np.clip(ecg_slice, -maxval, maxval)
        return ecg_slice
        
    else:
        return None
        