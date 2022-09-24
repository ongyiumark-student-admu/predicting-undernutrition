# Importing libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import imblearn as imb
from sklearn.model_selection import StratifiedKFold
from typing import Callable

# Creating aliases
PandasDataFrame = pd.DataFrame
TensorflowDataset = tf.data.Dataset
NumpyArrayPair = tuple[np.ndarray, np.ndarray]
ModelMetrics = tuple[float, float, float]

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

def get_metrics(predicted: np.ndarray, actual: np.ndarray) -> ModelMetrics:
    """
        Extracts metrics from predictions.

        :param predicted: numpy array of predictions
        :param actual: numpy array of ground truth
        :return (accuracy, sensitivity, specificity): model evaluation metrics
    """ 
    tp = np.sum((predicted==1)&(actual==1))
    tn = np.sum((predicted==0)&(actual==0))
    fp = np.sum((predicted==1)&(actual==0))
    fn = np.sum((predicted==0)&(actual==1))

    accuracy = (tp+tn)/(tp+tn+fp+fn)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    return accuracy, sensitivity, specificity

def train_dnn(train : PandasDataFrame, test : PandasDataFrame, label : str, features : list[str], layers : list[int]) -> np.ndarray:
    """
        Trains a dense neural network model with 'train' and evaluates the model on 'test'.

        :param train: pandas dataframe of the training set
        :param test: pandas dataframe of the testing set
        :param label: name of the target column for supervised learning
        :param features: list of features to include in training
        :param layers: list of number of nodes per layer
        :return predicted: numpy array of class predictions
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

    # Getting predictions
    output = np.asarray([x[0] for x in model.predict(test_ds)])
    predicted = np.where(output >= 0.5, 1, 0)

    return predicted

def train_naive_hive(train : PandasDataFrame, test : PandasDataFrame, label : str, num_hive : int, train_network : Callable, **kwargs) -> np.ndarray:
    """
        Trains a random hive by naively training neural networks of the same architecture on the whole training set.

        :param train: pandas dataframe of the training set
        :param test: pandas dataframe of the testing set
        :param label: name of the target column for supervised learning
        :param num_hive: number of networks in the hive
        :param train_network: training function for each network
        :param **kwargs: other keyword arguments for the training function
        :return predicted: numpy array of class predictions
    """

    # Training networks
    ballots = []
    for x in range(num_hive):
        print(f"Training network {x+1}...")
        ballots.append(train_network(train,test,label,**kwargs))
        print(f"Network {x+1} completed.")

    # Counting votes
    predicted = []
    for p in zip(*ballots):
        votes = sum(p)
        predicted.append(1 if 2*votes >= num_hive else 0)

    return np.asarray(predicted)

def train_kfold(train_set: PandasDataFrame, label: str, num_fold : int, to_smote: bool, train_func : Callable, **kwargs) -> dict:
    """
        Validates a model with stratified k-fold cross validation.

        :param train_set: pandas dataframe of the training set
        :param label: name of the target column for supervised learning
        :param num_fold: number of folds
        :param to_smote: flag for applying oversampling with SMOTE
        :param train_func: training function of the model being validated
        :param **kwargs: other keyword arguments for the training function
        :return metrics: dictionary of metrics including 'ACCURACY', 'SENSITIVITY', and 'SPECIFICITY'.
    """

    # Arrays for metrics
    acc_per_fold = []
    sens_per_fold = []
    spec_per_fold = []
    
    # Build K folds
    kfold = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=42)
    for train_idx, val_idx in kfold.split(train_set.drop(label, axis=1), train_set[[label]]):
        train = train_set.iloc[train_idx]
        if to_smote:
            x_train = train.drop([label, "idx"], axis = 1)
            y_train = train[[label]]
            smt = imb.over_sampling.SMOTE()
            x_train_res, y_train_res = smt.fit_resample(x_train, y_train)
            train = pd.merge(x_train_res, y_train_res, left_index=True, right_index=True)
        test = train_set.iloc[val_idx]

        # Train model
        predicted = train_func(train, test, label, **kwargs)
        accuracy, sensitivity, specificity = get_metrics(predicted, np.where(test[label].values == "INCREASED RISK", 1, 0))
        
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