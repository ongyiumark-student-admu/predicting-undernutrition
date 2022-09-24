from predunder.functions import df_to_nparray
import pandas as pd


def test_sample():
    df = pd.DataFrame({"a": [2, 4, 4], "b": [5, 2, 21]})
    X, y = df_to_nparray(df, 'b')
