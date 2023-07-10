## contributors:
## - Shawn O'Neil
## - GPT-4

import time
import pandas as pd
import numpy as np
import math
from pyspark.ml.linalg import Vector
import os
import psutil
from pyspark.sql import DataFrame
from typing import Generator



def loop_over_batches(df: DataFrame, 
                      max_rows_per_batch: int = 10000, 
                      verbose: bool = True) -> Generator[pd.DataFrame, None, None]:
    """
    Given a spark dataframe, produces a generator yielding pandas dataframes of a specified size
    over all the rows of the dataframe. This can be used to process the entire dataframe
    in batches, each converted to a pandas dataframe of a size that fits within the memory 
    of the driver. Example usage also showing feature vectorization:
    
    from n3cutils.ml.feature_engineering.vectorizer import vectorize
    from n3cutils.ml.training.batch_iterator import loop_over_batches
    from n3cutils.ml.feature_engineering.utils import cols_to_numpy_arrays

    df = vectorize(df,
                   boolean_cols = ["is_male", "diabetes_indicator"],
                   categorical_cols = ["race", "state", "data_partner_id"],
                   numeric_cols = ["age", "height", "bmi"],
                   date_cols = ["date_last_vaccine", "date_last_visit"],
                   keep_cols = ["target_outcome"]
                   )

    for pandas_df in loop_over_batches(df, max_rows_per_batch = 50000):
        np_arrays_dict = cols_to_numpy_arrays(pandas_df, ["vectorized_features", "target_outcome"])

        # np_arrays_dict["vectorized_features"] is a 2d numpy array
        # np_arrays_dict["target_outcome"] is 1d
        # do some work...

    Rows of the dataframe are distributed into partitions of roughly equal size, but it should not be
    assumed that each pandas_ds will have the same number of rows. 

    Generally, you will want to use large batch sizes to minimize data collection overhead. 
    The number of rows to specify will thus be a function of 1) the memory resources available, primarily
    on the driver but also on the executors where the conversion to pandas dataframes happens, and 2) the
    size and number of columns. Of course, your analysis or model will be competing for memory use as well.
    To help tune the batch size, by default `verbose = True` and rough memory usage statistics are reported
    in the logs as batches are processed. Note however that when the batch size is too large memory
    will be exhausted before logs can be generated. 

    IMPORTANT: This method reads each batch of the dataframe in serial, which is appropriate for
    applications supporting incremental training (such as deep learning) when spark-based parallelization 
    is not available and the total data will not fit in memory. For applications where chunks of data 
    can be analyzed independently and the results collected, spark's parallelization features should be used.
    """

    start_time = time.time()

    total_rows = df.count()
    num_batches = math.ceil(total_rows / max_rows_per_batch)

    if df.rdd.getNumPartitions() != num_batches:
        df = df.repartition(num_batches)

    rows_processed = 0
    partitions_processed = 0

    total_rows = df.count()
    num_partitions = df.rdd.getNumPartitions()

    if verbose:
        _log_it(f"Collecting first partition, dataframe total rows: {total_rows}, num partitions: {num_partitions}", start_time)

    columns = df.schema.fieldNames()
    for partition in df.rdd.mapPartitions(lambda iterator: [pd.DataFrame(list(iterator), columns=columns)]).toLocalIterator():
        rows_processed = rows_processed + partition.shape[0]
        partitions_processed = partitions_processed + 1

        yield partition

        if verbose:
            _log_it(f"Collected partition. Rows processed: {rows_processed} ({(100*rows_processed/total_rows):.2f}%), partitions_processed: {partitions_processed} ({(100 * partitions_processed/num_partitions):.2f}%), est. mem usage: {_get_memory_usage_percentage()}% used", start_time)


def _log_it(message, start_time):
    elapsed_time = time.time() - start_time

    # convert elapsed time to hours, minutes, and seconds
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # create a string in the hh:mm:ss.ss format
    elapsed_time_str = "{:02}:{:02}:{:05.2f}".format(int(hours), int(minutes), seconds)

    print(f"{elapsed_time_str} {message}")


def _get_memory_limit():
    with open('/sys/fs/cgroup/memory/memory.limit_in_bytes') as f:
        return int(f.read())

def _get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss  # in bytes

def _get_memory_usage_percentage():
    limit = _get_memory_limit()
    usage = _get_memory_usage()

    # calculate percentage and round to 2 decimal places
    percentage = round((usage / limit) * 100, 2)

    return percentage
