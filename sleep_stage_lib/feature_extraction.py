import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew, gmean

from sleep_stage_lib.filtering import bandpass_filter


def extract_freq_features(psd: np.ndarray, freqs: np.ndarray) -> dict:
    # normalize
    psd_norm = psd / (psd.sum() + 1e-10)
    idx_dom = np.argmax(psd)
    # cumulative for roll‑off
    cumsum = np.cumsum(psd_norm)
    rolloff_idx = np.searchsorted(cumsum, 0.85)
    return {
        'dominant_frequency': freqs[idx_dom],
        'spectral_entropy': -np.sum(psd_norm * np.log2(psd_norm + 1e-10)),
        'spectral_bandwidth': np.sqrt(np.sum(((freqs - freqs[idx_dom])**2) * psd_norm)),
        'spectral_centroid': np.sum(freqs * psd_norm),
        'spectral_rolloff': freqs[min(rolloff_idx, len(freqs)-1)],
        'spectral_flatness': gmean(psd + 1e-10) / (psd.mean() + 1e-10)
    }

def extract_features(signal: np.ndarray, fs: float) -> dict:
    # time‑domain
    feat = {
        'mean': signal.mean(),
        'std': signal.std(),
        'min': signal.min(),
        'max': signal.max(),
        'median': np.median(signal),
        'energy': np.sum(signal**2),
        'kurtosis': kurtosis(signal),
        'skewness': skew(signal)
    }
    # overall freq‑domain
    freqs, psd = welch(signal, fs, nperseg=min(256, len(signal)))
    feat.update(extract_freq_features(psd, freqs))
    return feat

def extract_multiband_features(signal: np.ndarray, fs: float, signal_column : str) -> dict:
    # define your five bands (Hz)
    if signal_column == 'eog':
        bands = {
            'b1': (0.5, 2.0),
            'b2': (2.0, 4.0),
            'b3': (4.0, 7.0),
            'b4': (7.0, 10.0),
            'b5': (10.0, 15.0),
        }
    elif signal_column == 'emg':
        # EMG bands within 20–99 Hz (five equal-width bands)
        bands= {
            'b1': (20.0, 36.0),
            'b2': (36.0, 52.0),
            'b3': (52.0, 68.0),
            'b4': (68.0, 84.0),
            'b5': (84.0, 99.0),
        }
    all_feat = {}
    # first, global features
    all_feat.update(extract_features(signal, fs))
    # then each band
    for name, (low, high) in bands.items():
        sig_f = bandpass_filter(signal, low, high, fs, order=4)
        sub_feat = extract_features(sig_f, fs)
        # prefix keys with band name
        for k, v in sub_feat.items():
            all_feat[f'{name}_{k}'] = v
    return all_feat

def extract_features_and_labels(df: pd.DataFrame,
                                signal_column: str,
                                label_column: str,
                                fs: float=200,
                                multiband = True) -> (pd.DataFrame, pd.Series):
    feats, labels = [], []
    for file_id, grp in df.groupby('file'):
        sig = grp[signal_column].values
        if multiband:
            feats.append(extract_multiband_features(sig, fs, signal_column))
        else:
            feats.append(extract_features(sig, fs))
        labels.append(grp[label_column].iat[0])
    return pd.DataFrame(feats), pd.Series(labels, name=label_column)

