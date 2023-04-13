import numpy as np

from predunder.functions import df_to_dataset, df_to_nparray, \
    get_metrics, oversample_data, undersample_data, kfold_metrics_to_df, convert_df_col_type
from predunder.training import train_dnn, train_kfold
from predunder.hypertuning import tune_dnn


def test_df_to_dataset(df_sample):
    df, label = df_sample
    df_to_dataset(df, label)


def test_df_to_nparray(df_sample):
    df, label = df_sample
    df_to_nparray(df, label)


def test_get_metrics():
    pred = np.where(np.random.rand(100) >= 0.5, 1, 0)
    actual = np.where(np.random.rand(100) >= 0.5, 1, 0)
    get_metrics(pred, actual)


def test_oversample_data(df_sample):
    df, label = df_sample
    oversample_data(df, label)
    oversample_data(df, label, "smote")
    oversample_data(df, label, "adasyn")
    oversample_data(df, label, "borderline")


def test_undersample_data(df_sample):
    df, label = df_sample
    undersample_data(df, label)
    undersample_data(df, label, "auto")
    undersample_data(df, label, 1.00)


def test_train_dnn(df_sample):
    df, label = df_sample
    train_dnn(df, df, label, [6])


def test_train_dnn_smote(df_sample):
    df, label = df_sample
    train_dnn(df, df, label, [6], 1, "smote")


def test_train_dnn_adasyn(df_sample):
    df, label = df_sample
    train_dnn(df, df, label, [6], 1, "adasyn")


def test_train_dnn_borderline(df_sample):
    df, label = df_sample
    train_dnn(df, df, label, [6], 1, "borderline")


def test_train_kfold(df_sample):
    df, label = df_sample
    train_kfold(df, label, 2, train_dnn, layers=[6])


def test_kfold_metrics_to_df(df_sample):
    df, label = df_sample
    metrics = train_kfold(df, label, 2, train_dnn, layers=[6])
    kfold_metrics_to_df(metrics)


def test_tune_dnn(df_sample):
    df, label = df_sample
    tune_dnn(df, label, 2, 1)


def test_convert_df_col_type(df_sample):
    df, label = df_sample
    columns = list(df.columns)
    new_types = list(df.dtypes)
    convert_df_col_type(df, columns, new_types)
