# [Logic Liaison Template] ML Utils: Feature Vectorization and Incremental Training in Python

*Shawn T. O'Neil, CU Anschutz*

## Background

Working with large datasets and machine learning tools in N3C, where data are primarily stored in Apache Spark
dataframes, can be challenging. While features are
naturally encoded as columns in dataframes, this format incurs significant performance costs for dataframes
with hundreds or thousands of columns, and full datasets are frequently too large to fit in memory
even with large-memory compute profiles.

Fortunately, Spark supports
more efficient data representation in numeric form with the `mllib` Vector type. Unfortunately,
converting many columns of various types (numeric, categorical, date, boolean) is tedious work;
here we describe pre-built functionality to accomplish this in a flexible way. 

Even when data are efficiently encoded, they may be too large to fit into memory for use in model
training in their entirety. However, some applications -- notably deep learning -- do not require
access to all data at once, but can instead operate on "batches" of data. Gathering and converting
these batches efficiently with Spark is also difficult, and we provide functionality to easily do so.
*Note: we have not yet tested this feature on a full, large-scale end-to-end deep learning or other
ML pipeline - if you do so please alert the authors of your use case!*

### Feature vectorization

For details on feature vectorization, see [Feature Vectorization README](https://unite.nih.gov/workspace/notepad/view/ri.notepad.main.notepad.f7589551-e3ba-4330-912c-d25422ae56c7) and the corresponding 
example [Feature Vectorization Example](https://unite.nih.gov/workspace/vector/view/ri.vector.main.workbook.4aacde1c-0116-4915-a647-ede949c2749b?branch=master).

### Incremental training

For details on incremental training, see [Incremental Training README](https://unite.nih.gov/workspace/notepad/view/ri.notepad.main.notepad.0784d9f6-818e-422c-a597-e5b5aee95637)
and the corresponding example [Incremental Training Example](https://unite.nih.gov/workspace/vector/view/ri.vector.main.workbook.8e753624-d137-4114-8bb3-e9f5e2530868?branch=master).


## Notes on implementation

These utilities are implemented only for Python, in a code repository (n3cutils).
The feature vectorization functionality however is very simple to use,both in code and
with the provided code workbook template. In theory the batch training functionality is
possible via SparkR, but this is not currently implemented.