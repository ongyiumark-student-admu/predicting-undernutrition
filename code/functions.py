# Importing libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from typing import Callable

# Creating aliases
PandasDataFrame = pd.DataFrame
TensorflowDataset = tf.data.Dataset
NumpyArrayPair = tuple[np.ndarray, np.ndarray]
ModelMetrics = tuple[float, float, float, float]

def df_to_dataset(dataframe: PandasDataFrame, label: str, shuffle: bool=True, batch_size: int=8) -> TensorflowDataset:
    """
        Creates a Tensorflow Dataset from a Pandas DataFrame.
        
        :param dataframe: pandas dataframe to be converted
        :param label: name of the target column for supervised learning
        :param shuffle: shuffles the dataset
        :param batch_size: batch size of the dataset
        :return tfdataset: tensorflow dataset based on the dataframe 
    """
    dataframe = dataframe.copy()
    dataframe['target'] = np.where(dataframe[label]=='INCREASED RISK', 1, 0)
    dataframe = dataframe.drop(columns=label)
    
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    tfdataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        tfdataset = tfdataset.shuffle(buffer_size=len(dataframe))
        tfdataset = tfdataset.batch(batch_size)
    return tfdataset

def df_to_nparray(dataframe: PandasDataFrame, label: str) -> NumpyArrayPair:
    """
        Converts the Pandas DataFrame into features and labels in the form of numpy arrays.
        
        :param dataframe: pandas dataframe to be converted
        :param label: name of the target column for supervised learning
        :return (X, y): features and labels for supervised learning 
    """
    X = dataframe.drop(label, axis=1).to_numpy()
    y = dataframe[label].to_numpy()
    return (X, y)

def train_nn(train : PandasDataFrame, test : PandasDataFrame, label : str, features : list[str], layers : list[int]) -> ModelMetrics:
    """
        Trains a dense neural network model with 'train' and evaluates the model on 'test'.

        :param train: pandas dataframe of the training set
        :param test: pandas dataframe of the testing set
        :param features: list of features to include in training
        :param label: name of the target column for supervised learning
        :param layers: list of number of nodes per layer
        :return (loss, accuracy, sensitivity, specificity): model evaluation metrics
    """
    # Generate feauture columns
    feature_columns = []
    for col in features:
        feature_columns.append(tf.feature_column.numeric_column(col))

    # Generating a tensorflow dataset
    train_ds = df_to_dataset(train, label)
    test_ds = df_to_dataset(test, label)

    # Building the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.DenseFeatures(feature_columns))
    for x in layers:
        model.add(tf.keras.layers.Dense(x, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Compiling the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.TruePositives(),
                           tf.keras.metrics.TrueNegatives(),
                           tf.keras.metrics.FalsePositives(),
                           tf.keras.metrics.FalseNegatives()
                          ])

    # Training the Model
    history = model.fit(train_ds, epochs=10, verbose=1)

    # Evaluating the Model
    scores = model.evaluate(test_ds, verbose=0)
    loss, accuracy, tp, tn, fp, fn = scores
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    
    return loss, accuracy, sensitivity, specificity

def train_kfold(train_set: PandasDataFrame, label: str, num_fold : int, train_func : Callable, **kwargs) -> dict:
    """
        Validates a model with stratified k-fold cross validation.

        :param train_set: pandas dataframe of the training set
        :param label: name of the target column for supervised learning
        :param num_fold: number of folds
        :param train_func: training function of the model being validated
        :param **kwargs: other keyword arguments for the training function
        :return metrics: dictionary of metrics including 'ACCURACY', 'SENSITIVITY', and 'SPECIFICITY'.
    """

    # Arrays for metrics
    acc_per_fold = []
    loss_per_fold = []
    sens_per_fold = []
    spec_per_fold = []
    
    # Build K folds
    kfold = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=42)
    for train_idx, val_idx in kfold.split(train_set.drop(label, axis=1), train_set[[label]]):
        train = train_set.iloc[train_idx] 
        test = train_set.iloc[val_idx]

        # Train model
        loss, accuracy, sensitivity, specificity = train_func(train, test, label, **kwargs)
        
        loss_per_fold.append(loss)
        acc_per_fold.append(accuracy)
        sens_per_fold.append(sensitivity)
        spec_per_fold.append(specificity)
    
    metrics = {
        'ACCURACY': {
            'ALL': acc_per_fold,
            'MEAN': np.mean(acc_per_fold),
            'STDEV': np.std(acc_per_fold)
        },
        'SENSITIVITY': {
            'ALL': sens_per_fold,
            'MEAN': np.mean(sens_per_fold),
            'STDEV': np.std(sens_per_fold)
        },
        'SPECIFICITY': {
            'ALL': spec_per_fold,
            'MEAN': np.mean(spec_per_fold),
            'STDEV': np.std(spec_per_fold)
        }
    }
    return metrics