import pytest
import pandas as pd


@pytest.fixture(scope="session")
def df_sample():
    task = '1b'
    train = pd.read_csv(f"train-test-data/{task}_train.csv", index_col=0)
    return train, task
