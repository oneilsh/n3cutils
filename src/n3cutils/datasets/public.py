from foundry.transforms import Dataset

# testing
def get_dataset(name):

    df = Dataset.get(name)

    return df
