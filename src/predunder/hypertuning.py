# Importing libraries
import pandas as pd

from sklearn.model_selection import ParameterGrid
from predunder.functions import kfold_metrics_to_df
from predunder.training import train_dnn, train_kfold


def tune_model(train, label, folds, train_func, param_grid, **kwargs):
    """Performs grid search cross validation on Cohen's Kappa

    :param train: DataFrame of the training set
    :type train: pandas.DataFrame
    :param label: name of the target column for supervised learning
    :type label: str
    :param fold: number of folds for k-fold cross-validation
    :type fold: int
    :param train_func: training function of the model being validated
    :type train_func: Callable[..., (float,float,float)]
    :param param_grid: grid of parameters to hypertune with
    :type param_grid: dict[str, list]
    :returns: best parameters
    :rtype: dict
    """

    best_score = -float('inf')
    all_params = list(ParameterGrid(param_grid))
    for i, params in enumerate(all_params):
        print(f"Starting parameters {i} of {len(all_params)}...")
        print(params)
        metrics = train_kfold(train, label, folds, train_func, **params, **kwargs)
        score = metrics['KAPPA']['MEAN']

        if best_score <= score:
            best_score = score
            best_params = params

        print(f"Completed parameters {i}: {score}.")
        print()

    return best_params


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

    for i in range(1, max_nodes+1):
        for _ in range(5):
            print()
        print("Training", [i])
        metrics = train_kfold(train, label, folds, train_dnn, layers=[i], oversample=oversample)
        rowdf = kfold_metrics_to_df(metrics)
        rowdf['LAYERS'] = [[i]]
        results = pd.concat([results, rowdf])

    for i in range(1, max_nodes+1):
        for j in range(1, max_nodes+1):
            for _ in range(5):
                print()
            print("Training", [i, j])
            metrics = train_kfold(train, label, folds, train_dnn, layers=[i, j], oversample=oversample)
            rowdf = kfold_metrics_to_df(metrics)
            rowdf['LAYERS'] = [[i, j]]
            results = pd.concat([results, rowdf])

    cols = results.columns.tolist()
    results = results[cols[-1:]+cols[:-1]]

    return results
