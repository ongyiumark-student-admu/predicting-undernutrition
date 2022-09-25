from predunder import functions as puf
import numpy as np


def test_df_to_dataset(df_sample):
    df, label = df_sample
    puf.df_to_dataset(df, label)


def test_df_to_nparray(df_sample):
    df, label = df_sample
    puf.df_to_nparray(df, label)


def test_get_metrics():
    pred = np.where(np.random.rand(100) >= 0.5, 1, 0)
    actual = np.where(np.random.rand(100) >= 0.5, 1, 0)
    puf.get_metrics(pred, actual)


def test_smote_data(df_sample):
    df, label = df_sample
    puf.smote_data(df, label)


def test_train_dnn(df_sample):
    df, label = df_sample
    puf.train_dnn(df, df, label, df.drop([label], axis=1).columns, [6])


def test_naive_hive(df_sample):
    df, label = df_sample
    puf.train_naive_hive(df, df, label, 2, puf.train_dnn, features=df.drop([label], axis=1).columns, layers=[6])


def test_train_kfold(df_sample):
    df, label = df_sample
    puf.train_kfold(df, label, 2, puf.train_dnn, features=df.drop([label], axis=1).columns, layers=[6])


def test_train_kfold_smote(df_sample):
    df, label = df_sample
    puf.train_kfold(df, label, 2, puf.train_dnn, True, features=df.drop([label], axis=1).columns, layers=[6])