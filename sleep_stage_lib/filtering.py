import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)
