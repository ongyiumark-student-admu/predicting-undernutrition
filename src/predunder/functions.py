# Importing libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import imblearn as imb
from PIL import Image, ImageDraw

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


def df_to_image(dataframe: PandasDataFrame, label: str, img_size: tuple[int, int], outdir: str) -> None:
    """
        Converts tabular data into images.

        :param dataframe: pandas dataframe to convert into image
        :param label: name of the target column for supervised learning
        :param img_size: dimensions of the resulting image
        :param outdir: directory where the images will be stored
    """

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    features = dataframe.drop([label], axis=1).columns.tolist()

    # Normalize variables
    normalize = ['IDD_SCORE', 'AGE', 'HHID_count', 'HH_AGE', 'FOOD_EXPENSE_WEEKLY',
                 'NON-FOOD_EXPENSE_WEEKLY', 'HDD_SCORE', 'FOOD_INSECURITY', 'YoungBoys', 'YoungGirls',
                 'AverageMonthlyIncome', 'FOOD_EXPENSE_WEEKLY_pc', 'NON-FOOD_EXPENSE_WEEKLY_pc',
                 'AverageMonthlyIncome_pc'
                 ]

    df_normal = dataframe.copy()
    for col in normalize:
        df_normal[col] = sigmoid((df_normal[col]-df_normal[col].mean())/df_normal[col].std())

    df_normal['BEN_4PS'] = df_normal['BEN_4PS']-1
    df_normal['label'] = np.where(df_normal['2aii'] == "INCREASED RISK", 1, 0)

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

        subdir = os.path.join(outdir, str(row['label']))
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        img.save(os.path.join(subdir, f'{index}.png'))


def image_to_dataset(dir: str, img_size: tuple[int, int]) -> TensorflowDataset:
    """
        Creates a Image Tensorflow Dataset from a directory.

        :param dir: directory of the images
        :param img_size: dimensions of the images
        :return dataset: tensorflow dataset based on the images
    """

    dataset = tf.keras.utils.image_dataset_from_directory(
        dir,
        shuffle=True,
        batch_size=8,
        image_size=img_size)
    return dataset


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
