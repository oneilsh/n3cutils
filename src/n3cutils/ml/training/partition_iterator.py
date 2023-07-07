import numpy as np
import pandas as pd
from pyspark.ml.linalg import VectorUDT

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader, random_split
# import time

# start_time = time.time()

def partitions_to_pandas_generator(df):
    num_partitions = df.rdd.getNumPartitions()

    # vector_columns = [f.name for f in df.schema.fields if isinstance(f.dataType, VectorUDT)]

    # print(vector_columns)

    # return None

    for partition_index in range(num_partitions):
        part_rdd = df.rdd.mapPartitionsWithIndex(lambda index, value: value if index == partition_index else iter([]))

        # collect the data from the current partition
        data_partitioned = part_rdd.collect()  # noqa

        # convert to pandas dataframe
        pd_df = pd.DataFrame(data_partitioned, columns=df.columns)

        yield pd_df




# def batch_train(prepped):
#     df = prepped.select("person_id", "confirmed_covid_patient", "scaled_features")
#     total_rows = df.count()

#     # initialize your model
#     input_dim = 387
#     #model = LogisticRegressionModel(input_dim)
#     model = DeepNN(input_dim)

#     # Loss function and optimizer
#     criterion = nn.BCELoss() # Binary Cross Entropy Loss for binary classification
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Stochastic Gradient Descent

#     num_partitions = df.rdd.getNumPartitions()

#     for macro_epoch in range(5):
#         for idx in range(num_partitions):
#             # filter data for the current partition
#             part_rdd = df.rdd.mapPartitionsWithIndex(lambda index, value: value if index == idx else iter([]))
            
#             # collect the data from the current partition
#             print(f"Collecting partition {idx}...")
#             data_partitioned = part_rdd.collect()

#             # convert to pandas dataframe
#             pd_df = pd.DataFrame(data_partitioned, columns=df.columns)
                        
#             # convert scaled_features to numpy array
#             pd_df['scaled_features'] = pd_df['scaled_features'].apply(lambda x: x.toArray() if x is not None else np.nan)

#             # Create a mask where any 'scaled_features' contains a NaN
#             mask = pd_df['scaled_features'].apply(lambda x: np.isnan(x).any() if isinstance(x, np.ndarray) else True)

#             # Use the mask to filter out the rows from your DataFrame
#             pd_df = pd_df[~mask]

#             # get the targets
#             pd_df['confirmed_covid_patient'] = pd_df['confirmed_covid_patient'].values

#             # Now drop rows with NaN in 'confirmed_covid_patient' column
#             pd_df = pd_df.dropna(subset=['confirmed_covid_patient'])

#             # Convert to numpy arrays after dropping NaNs
#             features_array = np.array(pd_df['scaled_features'].tolist())
#             target = pd_df['confirmed_covid_patient'].values


#             # get the person_id
#             id = pd_df['person_id'].values
            
#             # call your training function
#             train_macro_batch(id, model, target, features_array, macro_epoch, criterion, optimizer, total_rows)

#     return vectorized


# # Define the model
# class LogisticRegressionModel(nn.Module):
#     def __init__(self, input_dim):
#         super(LogisticRegressionModel, self).__init__()
#         self.linear = nn.Linear(input_dim, 1)
    
#     def forward(self, x):
#         outputs = torch.sigmoid(self.linear(x))
#         return outputs

# # Define the model
# class DeepNN(nn.Module):
#     def __init__(self, input_dim):
#         super(DeepNN, self).__init__()
#         self.layer1 = nn.Linear(input_dim, 100) # 100 nodes in the hidden layer 1
#         self.layer2 = nn.Linear(100, 50) # 50 nodes in the hidden layer 2
#         self.layer3 = nn.Linear(50, 1)
    
#     def forward(self, x):
#         x = torch.relu(self.layer1(x))
#         x = torch.relu(self.layer2(x))
#         outputs = torch.sigmoid(self.layer3(x))
#         return outputs


# def calculate_accuracy(outputs, targets):
#     predictions = outputs.round()
#     correct = (predictions == targets).float()
#     accuracy = correct.sum() / len(correct)
#     return accuracy


# # Training function
# def train_macro_batch(id, model, targets, features, macro_epoch, criterion, optimizer, total_rows):
#     targets = torch.from_numpy(targets.astype(np.float32))
#     features = torch.from_numpy(features.astype(np.float32))

#     # Create a TensorDataset and DataLoader for mini-batches
#     dataset = TensorDataset(features, targets)
#     train_size = int(0.8 * len(dataset))  # 80% of data for training
#     val_size = len(dataset) - train_size   # 20% of data for validation
#     train_data, val_data = random_split(dataset, [train_size, val_size])
#     train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
#     val_loader = DataLoader(val_data, batch_size=512)

#     # Training loop
#     for micro_epoch in range(1):  # 1 'mini epoch' (warning: setting higher than 1 may overfit on this macro batch)
#         model.train()  # set the model in training mode
#         minibatch_count = 0
#         records_count = 0

#         for features, targets in train_loader:
#             minibatch_count = minibatch_count + 1
#             records_count = records_count + len(targets)

#             optimizer.zero_grad()  # clear gradients

#             # forward pass
#             outputs = model(features)

#             # compute loss
#             loss = criterion(outputs, targets)

#             # backward pass and optimization
#             loss.backward()
#             optimizer.step()

#             if(minibatch_count % 100 == 0):
#                 # Evaluation loop on the validation set
#                 model.eval()  # set the model in evaluation mode
#                 with torch.no_grad():
#                     val_loss = 0
#                     val_acc = 0
#                     for features, targets in val_loader:
#                         outputs = model(features)
#                         val_loss += criterion(outputs, targets)
#                         val_acc += calculate_accuracy(outputs, targets)

#                 # Calculate average validation loss and accuracy
#                 val_loss = val_loss / len(val_loader)
#                 val_acc = val_acc / len(val_loader)

#                 # Print logs
#                 end_time = time.time()
#                 elapsed_time = end_time - start_time
#                 elapsed_hours, rem = divmod(elapsed_time, 3600)
#                 elapsed_minutes, elapsed_seconds = divmod(rem, 60)
#                 timestr = f"{elapsed_hours}:{elapsed_minutes}:{elapsed_seconds}"

#                 print(f'{timestr} Dataset Epoch: {macro_epoch}, Batch Epoch: {micro_epoch}, Records: {records_count} (of {total_rows}), Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
#                 model.train() # back to training mode

