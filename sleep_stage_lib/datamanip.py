from pathlib import Path
from typing import Callable, Any, List

import numpy as np
import pandas as pd

from sleep_stage_lib.config import PROCESSED_DATA_DIR

def import_app5_dataset() -> pd.DataFrame:
    input_path : Path = PROCESSED_DATA_DIR / "app5_dataset.csv"
    df = pd.read_csv(input_path)
    return df

def apply_filter_per_file(df: pd.DataFrame,
                          filter_func: Callable[..., np.ndarray],
                          column: str,
                          **filter_params: Any) -> pd.DataFrame:
    """
    Applies a given filter function to a specified column of the DataFrame
    for each file (grouped by the 'file' column).

    Parameters:
      df (pd.DataFrame): Input DataFrame with at least columns [column, 'file'].
      filter_func (Callable[..., np.ndarray]): Filtering function that accepts
            (data: np.ndarray, **filter_params) and returns np.ndarray.
      column (str): The column name to which the filter should be applied (e.g., 'eog' or 'emg').
      filter_params: Additional keyword arguments to pass to the filter function
                     (e.g., lowcut, highcut, fs, order).

    Returns:
      pd.DataFrame: DataFrame with the filtered values in the specified column.
    """
    filtered_groups: List[pd.DataFrame] = []

    # Process each file's data independently.
    for file_id, group in df.groupby('file'):
        group = group.copy()
        # Apply the filter function to the selected column
        group[column] = filter_func(group[column].values, **filter_params)
        filtered_groups.append(group)

    # Concatenate the filtered groups back into one DataFrame.
    return pd.concat(filtered_groups, ignore_index=True)