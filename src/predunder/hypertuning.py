# Importing libraries
import pandas as pd

from predunder.training import train_dnn, train_kfold
from predunder.functions import kfold_metrics_to_df

# Defining Aliases
PandasDataFrame = pd.core.frame.DataFrame


def tune_dnn(train: PandasDataFrame, label: str, folds: int, max_nodes: int, oversample: str = "none") -> PandasDataFrame:
    """
        Perfoms k-fold cross validation on dense neural networks with varying layers.

        :param train: pandas dataframe of the training set
        :param label: name of the target column for supervised learning
        :param fold: number of folds for k-fold cross-validation
        :param max_nodes: maximum number of nodes per layer
        :param oversample: oversampling algorithm to be applied
        :return results: pandas dataframe of results
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
