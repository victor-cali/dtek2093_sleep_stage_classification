import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew


def extract_features(signal, fs):
    # Time-domain features
    feat = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'median': np.median(signal),
        'energy': np.sum(signal ** 2),
        'kurtosis': kurtosis(signal),
        'skewness': skew(signal)
    }

    # Frequency-domain features using Welch's method
    freqs, psd = welch(signal, fs, nperseg=min(256, len(signal)))
    psd_norm = psd / (np.sum(psd) + 1e-10)  # Normalize PSD to sum to 1
    dominant_idx = np.argmax(psd)
    feat['dominant_frequency'] = freqs[dominant_idx]
    feat['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    feat['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - freqs[dominant_idx]) ** 2) * psd_norm))

    return feat


def extract_features_and_labels_from_dataset(df: pd.DataFrame, signal_column: str, label_column: str) -> tuple[
    pd.DataFrame, pd.DataFrame]:
    features = []
    labels = []

    for file_id, group in df.groupby('file'):
        # All rows in a file share the same label
        label = group[label_column].iloc[0]
        signal = group[signal_column].values
        feat = extract_features(signal, fs=200)
        features.append(feat)
        labels.append(label)

    return pd.DataFrame(features), pd.DataFrame(labels, columns=[label_column])
