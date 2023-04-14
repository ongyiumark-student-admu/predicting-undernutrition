import pandas as pd
import numpy as np

from predunder.functions import convert_labels, normalize, df_to_nparray, get_metrics


def test_covert_labels_two():
    two_labels = ['REDUCED RISK', 'INCREASED RISK']
    arr = np.random.randint(0, 2, size=100)
    narr = convert_labels([two_labels[x] for x in arr])
    assert all(a == b for a, b in zip(narr, arr))


def test_convert_labels_three():
    three_labels = ['UNDER', 'ADEQUATE', 'OVER']
    arr = np.random.randint(0, 3, size=100)
    narr = convert_labels([three_labels[x] for x in arr])
    assert all(a == b for a, b in zip(narr, arr))


def test_normalize():
    X_tr = np.array([[1, 2], [3, 6]])  # MEAN: [2, 4] and SD: [1, 2]
    X_te = np.array([[1, 3], [4, 5]])
    res_tr, res_te = normalize(X_tr, X_te)

    expected_tr = np.array([[-1., -1.], [1., 1.]])
    expected_te = np.array([[-1., -0.5], [2., 0.5]])
    assert np.all(expected_tr == res_tr)
    assert np.all(expected_te == res_te)


def test_df_to_nparray():
    data = {'a': [1, 2, 3], 'b': [1, 2, 5]}
    df = pd.DataFrame.from_dict(data)
    X, y = df_to_nparray(df, 'a')

    assert np.all(X == np.array([[1], [2], [5]]))
    assert np.all(y == np.array([1, 2, 3]))


def test_get_metrics():
    pred = np.array([1, 1, 1, 0, 0])
    actual = np.array([0, 1, 0, 1, 0])

    accuracy, sensitivity, specificity, kappa = get_metrics(pred, actual)
    assert accuracy == 2/5
    assert sensitivity == 1/2
    assert specificity == 1/3
    assert kappa == -0.15384615384615377
