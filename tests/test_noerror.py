import numpy as np
import os
from shutil import rmtree

from predunder.functions import df_to_dataset, df_to_nparray, df_to_image, image_to_dataset, \
    get_metrics, smote_data, kfold_metrics_to_df
from predunder.training import train_dnn, train_naive_hive, train_kfold
from predunder.hypertuning import tune_dnn


def test_df_to_dataset(df_sample):
    df, label = df_sample
    df_to_dataset(df, label)


def test_df_to_nparray(df_sample):
    df, label = df_sample
    df_to_nparray(df, label)


def test_df_to_image(df_sample):
    df, label = df_sample
    dir = "image-data/basic-sample"
    img_size = (160, 160)
    if os.path.exists(dir):
        rmtree(dir)

    df_to_image(df, label, img_size, dir)


def test_image_to_dataset(df_sample):
    df, label = df_sample
    dir = "image-data/basic-sample"
    if os.path.exists(dir):
        rmtree(dir)
    img_size = (160, 160)

    df_to_image(df, label, img_size, dir)
    image_to_dataset(dir, img_size)


def test_get_metrics():
    pred = np.where(np.random.rand(100) >= 0.5, 1, 0)
    actual = np.where(np.random.rand(100) >= 0.5, 1, 0)
    get_metrics(pred, actual)


def test_smote_data(df_sample):
    df, label = df_sample
    smote_data(df, label)


def test_train_dnn(df_sample):
    df, label = df_sample
    train_dnn(df, df, label, df.drop([label], axis=1).columns, [6])


def test_naive_hive(df_sample):
    df, label = df_sample
    train_naive_hive(df, df, label, 2, train_dnn, features=df.drop([label], axis=1).columns, layers=[6])


def test_train_kfold(df_sample):
    df, label = df_sample
    train_kfold(df, label, 2, train_dnn, features=df.drop([label], axis=1).columns, layers=[6])


def test_train_kfold_smote(df_sample):
    df, label = df_sample
    train_kfold(df, label, 2, train_dnn, True, features=df.drop([label], axis=1).columns, layers=[6])


def test_kfold_metrics_to_df(df_sample):
    df, label = df_sample
    metrics = train_kfold(df, label, 2, train_dnn, True, features=df.drop([label], axis=1).columns, layers=[6])
    kfold_metrics_to_df(metrics)


def test_tune_dnn(df_sample):
    df, label = df_sample
    tune_dnn(df, label, 2, 1, False)


def test_tune_dnn_smote(df_sample):
    df, label = df_sample
    tune_dnn(df, label, 2, 1, True)
