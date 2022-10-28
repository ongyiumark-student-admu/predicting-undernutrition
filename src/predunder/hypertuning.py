# Importing libraries
import pandas as pd

from predunder.functions import kfold_metrics_to_df
from predunder.training import train_dnn, train_kfold


def tune_dnn(train, label, folds, max_nodes, oversample="none"):
    """Brute force number of nodes in a neural network up to three hidden layers.

    :param train: DataFrame of the training set
    :type train: pandas.DataFrame
    :param label: name of the target column for supervised learning
    :type label: str
    :param fold: number of folds for k-fold cross-validation
    :type fold: int
    :param max_nodes: maximum number of nodes per layer
    :type max_nodes: int
    :param oversample: oversampling algorithm to be applied ("none", "smote", "adasyn", "borderline")
    :type oversample: str, optional
    :returns: DataFrame of results
    :rtype: pandas.DataFrame
    """
    results = pd.DataFrame()
    features = train.drop(
        [label], axis=1).columns

    for i in range(1, max_nodes+1):
        for _ in range(5):
            print()
        print("Training", [i])
        metrics = train_kfold(train, label, folds, train_dnn, features=features, layers=[i], oversample=oversample)
        rowdf = kfold_metrics_to_df(metrics)
        rowdf['LAYERS'] = [[i]]
        results = pd.concat([results, rowdf])

    for i in range(1, max_nodes+1):
        for j in range(1, max_nodes+1):
            for _ in range(5):
                print()
            print("Training", [i, j])
            metrics = train_kfold(train, label, folds, train_dnn, features=features, layers=[i, j], oversample=oversample)
            rowdf = kfold_metrics_to_df(metrics)
            rowdf['LAYERS'] = [[i, j]]
            results = pd.concat([results, rowdf])

    cols = results.columns.tolist()
    results = results[cols[-1:]+cols[:-1]]

    return results
