## contributors:
## - Shawn O'Neil
## - GPT-4

import numpy as np
from pyspark.ml.linalg import Vector
from pandas import DataFrame
from typing import List, Dict

def cols_to_numpy_arrays(pandas_df: DataFrame, cols: List[str]) -> Dict[str, np.ndarray]:
    """Given a df and cols expected to be numeric or vector, returns a dictionary of numpy arrays (1d or 2d) 
    keyed by column name."""

    arrays = {}

    for column in cols:
        if isinstance(pandas_df[column].iloc[0], Vector):
            array = np.vstack(pandas_df[column].apply(lambda x: x.toArray()))
            arrays[column] = array
        else:
            array = pandas_df[column].values
            arrays[column] = array

    return arrays
