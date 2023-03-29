# Importing libraries
import imblearn as imb
import numpy as np
import pandas as pd
import tensorflow as tf


def convert_labels(labels):
    """Converts labels to ordinal numbers.

    :param labels: array of labels for supervised learning
    :type labels: numpy.ndarray[str]
    :returns: converted array of labels
    :rtype: numpy.ndarray[int]
    """
    LABELS_DICT = {
        2: ['REDUCED RISK', 'INCREASED RISK'],
        3: ['UNDER', 'ADEQUATE', 'OVER']
    }
    sz = np.unique(labels).size
    return np.asarray(list(map(lambda v: LABELS_DICT[sz].index(v), labels)))


def df_to_dataset(dataframe, label, shuffle=True, batch_size=8):
    """Convert a Pandas DataFrame into a Tensorflow Dataset.

    :param dataframe: DataFrame to be converted
    :type dataframe: pandas.DataFrame
    :param label: name of the target column for supervised learning
    :type label: str
    :param shuffle: shuffles the dataset
    :type shuffle: bool, optional
    :param batch_size: batch size of the dataset
    :type batch_size: int, optional
    :returns: Tensorflow Dataset based on the DataFrame
    :rtype: tensorflow.data.Dataset

    .. note:: The dataframe is required to have this as one of its columns.
    """
    dataframe = dataframe.copy()
    dataframe['target'] = convert_labels(dataframe[label].values)
    dataframe = dataframe.drop(columns=label)

    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    tfdataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        tfdataset = tfdataset.shuffle(buffer_size=len(dataframe))
    tfdataset = tfdataset.batch(batch_size)
    return tfdataset


def df_to_nparray(dataframe, label):
    """Split a Pandas DataFrame into features and labels.

    :param dataframe: DataFrame to be converted
    :type dataframe: pandas.DataFrame
    :param label: name of the target column for supervised learning
    :returns: features and labels for supervised learning
    :rtype: (numpy.ndarray, numpy.ndarray)

    .. note:: The dataframe is required to have this as one of its columns.
    .. todo:: This is currently not being used anywhere.
    """
    X = dataframe.drop(label, axis=1).to_numpy()
    y = dataframe[label].to_numpy()

    return (X, y)


def get_metrics(predicted, actual):
    """Extract relevant metrics from predictions.

    :param predicted: array of predictions
    :type predicted: numpy.ndarray
    :param actual: array of ground truth
    :type actual: numpy.ndarray
    :returns: model evaluation metrics (accuracy, sensitivity, specificity)
    :rtype: (float, float, float)

    .. todo:: This currently does not have Cohen's Kappa.
    .. todo:: This only supports binary classification.
    """
    tp = np.sum((predicted == 1) & (actual == 1))
    tn = np.sum((predicted == 0) & (actual == 0))
    fp = np.sum((predicted == 1) & (actual == 0))
    fn = np.sum((predicted == 0) & (actual == 1))

    accuracy = (tp+tn)/(tp+tn+fp+fn)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    po = (tp+tn)/(tp+tn+fp+fn)
    pe = ((tp+fn)*(tp+fp) + (fp+tn)*(fn+tn))/(tp+tn+fp+fn)**2
    kappa = (po-pe)/(1-pe)

    return accuracy, sensitivity, specificity, kappa


def kfold_metrics_to_df(metrics, include_all=False, include_stdev=True):
    """Convert a k-fold metrics dictionary into a Pandas DataFrame row for analysis.

    :param metrics: dictionary of metrics from predunder.training.train_kfold(..)
    :type metrics: dict
    :param include_all: includes a list of all the metrics per fold
    :type include_all: bool, optional
    :param include_stdev: includes the standard devation of metrics across folds
    :type include_stdev: bool, optional
    :returns: DataFrame with a single row
    :rtype: pandas.DataFrame
    """

    dfrow = pd.DataFrame()
    for metric, vals in metrics.items():
        for key, val in vals.items():
            dfrow[f"{metric}_{key}"] = [val]

    if not include_all:
        dfrow = dfrow.drop(dfrow.filter(regex='ALL$').columns.to_list(), axis=1)

    if not include_stdev:
        dfrow = dfrow.drop(dfrow.filter(regex='STDEV$').columns.to_list(), axis=1)

    return dfrow


def convert_df_col_type(dataframe, columns, new_types="int64"):
    """Convert the specified columns in a Pandas DataFrame into the desired datatypes.

    :param dataframe: DataFrame to be modified
    :type dataframe: pandas.DataFrame
    :param columns: list of names of columns to be converted
    :type columns: list
    :param new_types: list of desired datatypes or single desired datatype
    :type new_types: list or str, optional
    :returns: DataFrame with converted columns
    :rtype: pandas.DataFrame

    .. todo:: Add support for column indices instead of names, maybe.
    .. todo:: Add support for dictionary input that combines columns and new_types.
    """

    if type(new_types) == str:
        new_types = [new_types]*len(columns)

    dtype = dict(zip(columns, new_types))
    converted_dataframe = dataframe.astype(dtype)

    return converted_dataframe


def oversample_data(train_set, label, oversample="none"):
    """Perform oversampling over the minority class using the specified technique.

    :param train_set: DataFrame of the training set
    :type train_set: pandas.DataFrame
    :param label: name of the target column for supervised learning
    :type label: str
    :param oversample: oversampling algorithm to be applied ("none", "smote", "adasyn", "borderline")
    :type oversample: str, optional
    :returns: DataFrame of training set with oversampled data
    :rtype: pandas.DataFrame
    """

    if oversample == "none":
        return train_set

    x_train = train_set.drop([label], axis=1)
    y_train = train_set[[label]]

    if oversample == "smote":
        ovs = imb.over_sampling.SMOTE()
    elif oversample == "adasyn":
        ovs = imb.over_sampling.ADASYN()
    elif oversample == "borderline":
        ovs = imb.over_sampling.BorderlineSMOTE()
    x_train, y_train = ovs.fit_resample(x_train, y_train)
    train = pd.merge(x_train, y_train, left_index=True, right_index=True)

    return train
