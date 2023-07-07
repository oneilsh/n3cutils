from foundry.transforms import Dataset


def get_dataset(name):

    df = Dataset.get(name)

    return df
