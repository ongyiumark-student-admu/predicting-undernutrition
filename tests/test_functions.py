import pandas as pd
import numpy as np

from predunder.functions import convert_labels, normalize, df_to_nparray, get_metrics, \
    kfold_metrics_to_df


def test_covert_labels_two():
    two_labels = ['REDUCED RISK', 'INCREASED RISK']
    arr = np.random.randint(0, 2, size=100)
    narr = convert_labels([two_labels[x] for x in arr])
    assert all(a==b for a, b in zip(narr, arr))


def test_convert_labels_three():
    three_labels = ['UNDER', 'ADEQUATE', 'OVER']
    arr = np.random.randint(0, 3, size=100)
    narr = convert_labels([three_labels[x] for x in arr])
    assert all(a==b for a, b in zip(narr, arr))
    

def test_normalize():
    X_tr = np.array([[1, 2], [3, 6]]) # MEAN: [2, 4] and SD: [1, 2]
    X_te = np.array([[1, 3], [4, 5]])
    res_tr, res_te = normalize(X_tr, X_te)

    expected_tr = np.array([[-1., -1.], [1., 1.]])
    expected_te = np.array([[-1., -0.5], [2., 0.5]])
    assert np.all(expected_tr == res_tr)
    assert np.all(expected_te == res_te)
