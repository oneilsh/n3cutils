from pyspark.ml.feature import OneHotEncoder, StringIndexer, StandardScaler, VectorAssembler
from pyspark.sql.functions import col, datediff, lit, concat, concat_ws, mean, stddev, when, to_date, regexp_replace, udf, to_date
from pyspark.sql.types import StringType, ArrayType, IntegerType
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
import math

def vectorize(df, boolean_cols = [], 
                  continuous_cols = [],
                  date_cols = [],
                  categorical_cols = [],
                  vector_cols = [],
                  keep_cols = [],
                  id_col = "person_id",
                  output_prefix = "vectorized",
                  scale = True,
                  scale_vectors = False,
                  drop = True,
                  drop_unused = False,
                  batch_size = 0):

    keep_df = df.select(id_col, *keep_cols)


    # lookup vector name columns if they exist
    vector_names_cols = []
    for vector_col in vector_cols:
        if vector_col + "_names" in df.columns:
            vector_names_cols.append(vector_col + "_names")

    if len(boolean_cols) > 0:
        df = vectorize_booleans(df, boolean_cols, output_prefix + "_features", drop)

    if len(continuous_cols) > 0:
        df = vectorize_continuous(df, continuous_cols, output_prefix + "_features", scale, drop)

    if len(date_cols) > 0:
        df = vectorize_dates(df, date_cols, "2018-01-01", output_prefix + "_features", scale, drop)

    if len(categorical_cols) > 0:
        df = vectorize_categoricals(df, categorical_cols, output_prefix + "_features", drop)

    if len(vector_cols) > 0:
        df = vectorize_vectors(df, vector_cols, output_prefix + "_features", scale_vectors, drop)

    if drop_unused:
        df = df.select(id_col, output_prefix + "_features", output_prefix + "_features_names")

    if len(keep_cols) > 0:
        df = df.join(keep_df, id_col)

    if batch_size > 0:
        total_rows = df.count()
        num_batches = math.ceil(total_rows / batch_size)  # Total number of rows divided by your batch size, rounded up

        # Repartition the DataFrame based on the batch size
        df = df.repartition(num_batches)


    return df



def vectorize_vectors(df, 
                      vector_cols = [],
                      merge_into = "vectorized_features",
                      scale = False,
                      drop = True):

    merge_into_names = merge_into + "_names"

    if len(vector_cols) == 0:
        raise ValueError("No vector columns given to vectorize!")


    # Create a VectorAssembler to merge all the transformed columns into a single vector column
    assembler = VectorAssembler(
        inputCols=vector_cols,
        outputCol="temp_vector_features",
        handleInvalid = "skip"
    )
    df = assembler.transform(df)

    if scale:
        scaler = StandardScaler(
            inputCol = "temp_vector_features",
            outputCol = "temp_scaled_vector_features",
            withMean = True,
            withStd = True
        )
        df = scaler.fit(df).transform(df)
        df = df.drop("temp_vector_features").withColumnRenamed("temp_scaled_vector_features", "temp_vector_features")


    df = do_merge_vectors(df, merge_into, "temp_vector_features")


    ## For vector columns, e.g. some_features, we look for an existing names column, e.g. some_features_names.
    ## if it doesn't exist, we construct it
    ## since the vectors are added in with continuous data above, we add in the vector names to the continuous names
    size_udf = udf(lambda vector: len(vector), IntegerType())

    vector_cols_names = []
    for vector_col in vector_cols:
        first_vector_size = df.select(size_udf(df[vector_col])).first()[0]
        names_col = vector_col + "_names"
        if names_col in df.columns:
            vector_cols_names.append(names_col)
        else:
            vec_size = df.select(size_udf(df[vector_col])).first()[0]
            vec_names = []
            for i in range(vec_size):
                vec_names.append('"' + vector_col + '[' + str(i) + ']"')

            df = df.withColumn(names_col, lit(",".join(vec_names)))
            vector_cols_names.append(names_col)

    df = df.withColumn("temp_vector_feature_names", concat_ws(",", *vector_cols_names))

    if merge_into_names not in df.columns:
        df = df.withColumnRenamed("temp_vector_feature_names", merge_into_names)
    else:
        df = df.withColumn("temp_vector_names_all", concat_ws(",", merge_into_names, "temp_vector_feature_names"))
        df = df.drop(merge_into_names, "temp_vector_feature_names")
        df = df.withColumnRenamed("temp_vector_names_all", merge_into_names)

    df = df.drop("temp_vector_names_all", *[col for col in df.columns if col.endswith("_temp_names")])

    if drop:
        df = df.drop(*vector_cols, *vector_cols_names)

    return df


def vectorize_continuous(df,
                         continuous_cols = [],
                         merge_into = "vectorized_features",
                         scale = True,
                         drop = True):

    merge_into_names = merge_into + "_names"

    if len(continuous_cols) == 0:
        raise ValueError("No continuous columns given to vectorize!")

    

    # Create a VectorAssembler to merge all the transformed columns into a single vector column
    assembler = VectorAssembler(
        inputCols=continuous_cols,
        outputCol="temp_continuous_features",
        handleInvalid = "skip"
    )
    df = assembler.transform(df)

    if scale:
        scaler = StandardScaler(
            inputCol = "temp_continuous_features",
            outputCol = "temp_scaled_continuous_features",
            withMean = True,
            withStd = True
        )
        df = scaler.fit(df).transform(df)
        df = df.drop("temp_continuous_features").withColumnRenamed("temp_scaled_continuous_features", "temp_continuous_features")


    df = do_merge_vectors(df, merge_into, "temp_continuous_features")

    ### add names
    feature_names = ",".join(['"' + col + '"' for col in (continuous_cols)])
    df = add_string_to_column(df, merge_into_names, feature_names)

    if drop:
        df = df.drop(*continuous_cols)

    return df

def vectorize_dates(df,
                    date_cols = [],
                    ref_date = "2018-01-01",
                    merge_into = "vectorized_features",
                    scale = True,
                    drop = True):


    merge_into_names = merge_into + "_names"

    if len(date_cols) == 0:
        raise ValueError("No date columns given to vectorize!")

    # Convert date columns to numeric representation (days since reference date)
    reference_date = to_date(lit(ref_date))
    date_cols_transformed = [
        datediff(to_date(col), reference_date).alias(col + "_temp_datediff") for col in date_cols
    ]

    # we need to add in these new columns via this strange select; the datediff above redefines
    # each date column as an integer, but we have to run them through select (delayed eval...?)
    df = df.select(*df.columns, *date_cols_transformed)


    # Create a VectorAssembler to merge all the transformed columns into a single vector column
    assembler = VectorAssembler(
        inputCols=[col + "_temp_datediff" for col in date_cols],
        outputCol="temp_date_features",
        handleInvalid = "skip"
    )
    df = assembler.transform(df)

    ## scale if asked
    if scale:
        scaler = StandardScaler(
            inputCol = "temp_date_features",
            outputCol = "temp_scaled_date_features",
            withMean = True,
            withStd = True
        )
        df = scaler.fit(df).transform(df)
        df = df.drop("temp_date_features").withColumnRenamed("temp_scaled_date_features", "temp_date_features")

    df = do_merge_vectors(df, merge_into, "temp_date_features")

    ### add names
    feature_names = ",".join(['"' + col + '"' for col in (date_cols)])
    df = add_string_to_column(df, merge_into_names, feature_names)

    ## cleanup
    to_drop = [col for col in df.columns if col.endswith("_temp_datediff")]
    df = df.drop(*to_drop)

    ## drop source cols if asked
    if drop:
        df = df.drop(*date_cols)


    return df



def vectorize_categoricals(df,
                           categorical_cols = [],
                           merge_into = "vectorized_features",
                           drop = True):

    merge_into_names = merge_into + "_names"

    if len(categorical_cols) == 0:
        raise ValueError("No categorical columns given to vectorize!")


    # Perform one-hot encoding on categorical columns
    # skip nulls during indexing...
    string_indexers = [
        StringIndexer(inputCol=col, outputCol="{}_temp_indexed".format(col), handleInvalid = "skip")
        for col in categorical_cols
    ]
    
    # Fit StringIndexers to data
    fitted_indexers = [indexer.fit(df) for indexer in string_indexers]
    
    # Transform data with fitted StringIndexers
    for indexer in fitted_indexers:
        df = indexer.transform(df)


    # create the encoders
    one_hot_encoders = [
        # ... error if any non-allowed entries found (won't be since we skipped nulls above)
        OneHotEncoder(inputCol="{}_temp_indexed".format(col), outputCol="{}_temp_onehot".format(col), handleInvalid = "error", dropLast = False) # Nulls are all 0s
        for col in categorical_cols
    ]

    df = Pipeline(stages=one_hot_encoders).fit(df).transform(df)

    ## assemble!
    onehot_assembler = VectorAssembler(
        inputCols=[col for col in df.columns if col.endswith("_temp_onehot")],
        outputCol="temp_categorical_features",
        handleInvalid = "skip"
    )

    df = onehot_assembler.transform(df)
    df = do_merge_vectors(df, merge_into, "temp_categorical_features")

    # Add feature names
    feature_names_list = []
    for indexer in fitted_indexers:
        feature_names = ",".join(['"' + indexer.getOutputCol().replace('_temp_indexed', '') + "_" + label + '"' for label in indexer.labels])
        feature_names_list.append(feature_names)

    df = add_string_to_column(df, merge_into_names, ",".join(feature_names_list), ",")

    to_drop = [col for col in df.columns if col.endswith("_temp_indexed") or col.endswith("temp_onehot")]
    print(to_drop)
    df = df.drop(*to_drop)

    if drop:
        df = df.drop(*categorical_cols)

    return df





def vectorize_booleans(df, 
                       boolean_cols = [], 
                       merge_into = "vectorized_features",
                       drop = True):

    merge_into_names = merge_into + "_names"

    if len(boolean_cols) == 0:
        raise ValueError("No boolean columns given to vectorize!")

    assembler = VectorAssembler(
        inputCols=boolean_cols,
        outputCol="boolean_features",
        handleInvalid = "skip"
    )

    df = assembler.transform(df)
    feature_names = ",".join(['"' + col + '"' for col in (boolean_cols)])

    df = add_string_to_column(df, merge_into_names, feature_names)
    df = do_merge_vectors(df, merge_into, "boolean_features")
    
    if drop:
        df = df.drop(*boolean_cols)

    return df



####### utlity functions

def add_string_to_column(df, colname, string, sep = ","):
    if colname not in df.columns:
        df = df.withColumn(colname, lit(string))
        return df
    
    df = df.withColumn("temp_col_lkjlkj", concat_ws(sep, col(colname), lit(string)))
    df = df.drop(colname).withColumnRenamed("temp_col_lkjlkj", colname)

    return df


def do_merge_vectors(df, merge_into, new_vec_colname):
    if merge_into not in df.columns:
        df = df.withColumnRenamed(new_vec_colname, merge_into)
        return df

    assembler = VectorAssembler(
        inputCols = [merge_into, new_vec_colname],
        outputCol = "temp_col_slkdjf",
        handleInvalid = "skip"
    )

    df = (assembler.transform(df)
            .drop(merge_into)
            .withColumnRenamed("temp_col_slkdjf", merge_into)
            .drop(new_vec_colname)
         )

    return df

