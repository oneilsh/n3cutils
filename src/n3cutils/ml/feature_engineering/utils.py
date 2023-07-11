## contributors:
## - Shawn O'Neil
## - GPT-4

import numpy as np
from pyspark.ml.linalg import Vector
from pandas import DataFrame
from typing import List, Dict

def pd_cols_to_numpy_arrays(pandas_df: DataFrame, cols: List[str]) -> Dict[str, np.ndarray]:
    """Given a df and cols expected to hold numpy arrays, returns a dictionary of numpy arrays (1d or 2d) created by stacking rows, 
    keyed by column name."""

    arrays = {}

    for column in cols:
        arrays[column] = np.vstack(pandas_df[column])
    return arrays
