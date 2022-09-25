import pytest
import pandas as pd


@pytest.fixture(scope="session")
def df_sample():
    train = pd.read_csv("train-test-data/2aii_train.csv", index_col=0)
    return train, '2aii'
