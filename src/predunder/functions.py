# Importing libraries
import os

import imblearn as imb
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw


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


def df_to_image(dataframe, mean_df, std_df, label, img_size, out_dir):
    """Convert a Pandas DataFrame into a directory of images without regard for pixel position.

    :param dataframe: DataFrame to convert into images
    :type dataframe: pandas.DataFrame
    :param mean_df: DataFrame of column means for normalization
    :type mean_df: pandas.DataFrame
    :param std_df: DataFrame of column standard deviations for normalization
    :type std_df: pandas.DataFrame
    :param label: name of the target column for supervised learning
    :type label: str
    :param img_size: dimensions of the resulting images
    :type img_size: (int, int)
    :param out_dir: output directory where the images will be stored
    :type out_dir: str

    .. todo:: Categorical variables are hard-coded.
    """

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    features = dataframe.drop([label], axis=1).columns.tolist()

    # Normalize variables
    normalize = ['AGE', 'HHID_count', 'HH_AGE', 'FOOD_EXPENSE_WEEKLY',
                 'NON-FOOD_EXPENSE_WEEKLY', 'YoungBoys', 'YoungGirls',
                 'AverageMonthlyIncome', 'FOOD_EXPENSE_WEEKLY_pc', 'NON-FOOD_EXPENSE_WEEKLY_pc',
                 'AverageMonthlyIncome_pc'
                 ]

    df_normal = dataframe.copy()
    for col in normalize:
        df_normal[col] = sigmoid((df_normal[col]-mean_df[col])/std_df[col])

    df_normal['CHILD_SEX'] = df_normal['CHILD_SEX']/1
    df_normal['IDD_SCORE'] = df_normal['IDD_SCORE']/12
    df_normal['HDD_SCORE'] = df_normal['HDD_SCORE']/12
    df_normal['FOOD_INSECURITY'] = (df_normal['FOOD_INSECURITY']-1)/3
    df_normal['BEN_4PS'] = df_normal['BEN_4PS']/2
    df_normal['AREA_TYPE'] = df_normal['AREA_TYPE']/1

    df_normal[label] = convert_labels(df_normal[label].values)

    # Generate images
    n = len(features)
    w, h = img_size
    nw = n//4
    nh = (n+nw-1)//nw

    for index, row in df_normal.iterrows():
        img = Image.new("RGB", img_size)
        for i in range(0, nh):
            for j in range(0, nw):
                idx = i*nw+j
                if idx >= n:
                    break
                val = int(sigmoid(row[features[idx]])*255)

                r = ImageDraw.Draw(img)
                x = i*(h//nh)
                y = j*(w//nw)
                r.rectangle([(y, x), (y+w//nw, x+h//nh)], fill=(val, val, val))

        subdir = os.path.join(out_dir, str(row[label]))
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        img.save(os.path.join(subdir, f'{index}.png'))


def image_to_dataset(dir, img_size):
    """Create a Tensorflow Dataset from a directory of images.

    :param dir: directory of the images
    :type dir: str
    :param img_size: dimension of the images
    :type img_size: (int, int)
    :returns: Tensorflow Dataset based on the images
    :rtype: tensorflow.data.Dataset

    .. todo:: The batch size is fixed to 32.
    """

    dataset = tf.keras.utils.image_dataset_from_directory(
        dir,
        shuffle=True,
        batch_size=32,
        image_size=img_size)
    return dataset


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

    return accuracy, sensitivity, specificity


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


def oversample_data(train_set, label, oversample="none"):
    """Perform oversampling over the minority class using the specific technique.

    :param train_set: DataFrame of the training set
    :type train_set: pandas.DataFrame
    :param label: name of the target column for supervised learning
    :type label: str
    :param oversample: oversampling algorithm to be applied ("none", "smote", "adasyn", "borderline")
    :type oversample: str, optional
    :returns: DataFrame of training set with oversampled data
    :rtype: pandas.DataFrame
    """

    # Returns origin train set if no oversampling is specified
    if oversample == "none":
        return train_set

    # Separating dataset into features and labels
    x_train = train_set.drop([label], axis=1)
    y_train = train_set[[label]]

    # Applying specified oversampling techinque
    if oversample == "smote":
        ovs = imb.over_sampling.SMOTE()
    elif oversample == "adasyn":
        ovs = imb.over_sampling.ADASYN()
    elif oversample == "borderline":
        ovs = imb.over_sampling.BorderlineSMOTE()
    x_train, y_train = ovs.fit_resample(x_train, y_train)
    train = pd.merge(x_train, y_train, left_index=True, right_index=True)

    return train
