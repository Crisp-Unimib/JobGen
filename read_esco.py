import pandas as pd


def read_and_prepare_datasets(path):

    return pd.read_csv(path + "/data/ojas_to_generate.csv"), pd.read_csv(path + "/data/oja_examples.csv")

