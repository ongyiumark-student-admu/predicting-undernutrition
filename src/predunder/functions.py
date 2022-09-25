# Importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import imblearn as imb

from typing import Any
import numpy.typing as npt

# Defining Aliases
PandasDataFrame = pd.core.frame.DataFrame
TensorflowDataset = Any
FeatureLabelPair = tuple[npt.NDArray[np.float64], npt.NDArray[np.unicode_]]
ModelMetrics = tuple[float, float, float]


def df_to_dataset(dataframe: PandasDataFrame, label: str, shuffle: bool = True, batch_size: int = 8) -> TensorflowDataset:
    """
        Creates a Tensorflow Dataset from a Pandas DataFrame.

        :param dataframe: pandas dataframe to be converted
        :param label: name of the target column for supervised learning
        :param shuffle: shuffles the dataset
        :param batch_size: batch size of the dataset
        :return tfdataset: tensorflow dataset based on the dataframe
    """
    dataframe = dataframe.copy()
    dataframe['target'] = np.where(dataframe[label] == 'INCREASED RISK', 1, 0)
    dataframe = dataframe.drop(columns=label)

    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    tfdataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        tfdataset = tfdataset.shuffle(buffer_size=len(dataframe))
        tfdataset = tfdataset.batch(batch_size)
    return tfdataset


def df_to_nparray(dataframe: PandasDataFrame, label: str) -> FeatureLabelPair:
    """
        Converts the Pandas DataFrame into features and labels in the form of numpy arrays.

        :param dataframe: pandas dataframe to be converted
        :param label: name of the target column for supervised learning
        :return (X, y): features and labels for supervised learning
    """
    X = dataframe.drop(label, axis=1).to_numpy()
    y = dataframe[label].to_numpy()
    return (X, y)


def get_metrics(predicted: npt.NDArray[np.int64], actual: npt.NDArray[np.int64]) -> ModelMetrics:
    """
        Extracts metrics from predictions.

        :param predicted: numpy array of predictions
        :param actual: numpy array of ground truth
        :return (accuracy, sensitivity, specificity): model evaluation metrics
    """
    tp = np.sum((predicted == 1) & (actual == 1))
    tn = np.sum((predicted == 0) & (actual == 0))
    fp = np.sum((predicted == 1) & (actual == 0))
    fn = np.sum((predicted == 0) & (actual == 1))

    accuracy = (tp+tn)/(tp+tn+fp+fn)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    return accuracy, sensitivity, specificity


def kfold_metrics_to_df(metrics: dict) -> PandasDataFrame:
    """
        Converts k-fold metrics to a pandas dataframe for analysis.

        :param metrics: dictionary of metrics
        :return dfrow: a pandas dataframe with a single row
    """

    dfrow = pd.DataFrame()
    for metric, vals in metrics.items():
        for key, val in vals.items():
            dfrow[f"{metric}_{key}"] = [val]
    return dfrow


def smote_data(train_set: PandasDataFrame, label: str) -> PandasDataFrame:
    """
        Performs basic SMOTE over the training set.

        :param train_set: pandas dataframe of the training set
        :param label: name of the target column for supervised learning
        :return train: pandas dataframe of training set with oversampled data
    """
    x_train = train_set.drop([label], axis=1)
    y_train = train_set[[label]]
    smt = imb.over_sampling.SMOTE()
    x_train_sm, y_train_sm = smt.fit_resample(x_train, y_train)
    train = pd.merge(x_train_sm, y_train_sm, left_index=True, right_index=True)

    return train
