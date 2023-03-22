# Importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from xgboost import XGBClassifier
from nnrf import NNRF
from nnrf import ml

from predunder.functions import (convert_labels, df_to_nparray, df_to_dataset, get_metrics, oversample_data)


def train_random_forest(train, test, label, oversample="none", to_normalize=False, **kwargs):
    """Train a random forest model and make predictions.

    :param train: DataFrame of the training set
    :type train: pandas.DataFrame
    :param test: DataFrame of the testing set
    :type test: pandas.DataFrame
    :param label: name of the target column for supervised learning
    :type label: str
    :param oversample: oversampling algorithm to be applied ("none", "smote", "adasyn", "borderline")
    :type oversample: str, optional
    :param to_normalize: normalize features
    :type to_normalize: bool, optional
    :returns: array of class predictions
    :rtype: np.ndarray[int]
    """
    # Oversampling the training set
    train = oversample_data(train, label, oversample)

    clf = RandomForestClassifier(random_state=42, **kwargs)
    X_train, y_train = df_to_nparray(train, label)
    X_test, y_test = df_to_nparray(test, label)

    # normalize features
    if to_normalize:
        X_train = normalize(X_train)
        X_test = normalize(X_test)

    clf.fit(X_train, convert_labels(y_train))
    predicted = clf.predict(X_test)
    return predicted


def train_xgboost(train, test, label, oversample="none", to_normalize=False, **kwargs):
    """Train an XGBoost model and make predictions.

    :param train: DataFrame of the training set
    :type train: pandas.DataFrame
    :param test: DataFrame of the testing set
    :type test: pandas.DataFrame
    :param label: name of the target column for supervised learning
    :type label: str
    :param oversample: oversampling algorithm to be applied ("none", "smote", "adasyn", "borderline")
    :type oversample: str, optional
    :param to_normalize: normalize features
    :type to_normalize: bool, optional
    :returns: array of class predictions
    :rtype: np.ndarray[int]
    """
    # Oversampling the training set
    train = oversample_data(train, label, oversample)

    clf = XGBClassifier(random_state=42, objective='binary:logistic', **kwargs)
    X_train, y_train = df_to_nparray(train, label)
    X_test, y_test = df_to_nparray(test, label)

    # normalize features
    if to_normalize:
        X_train = normalize(X_train)
        X_test = normalize(X_test)

    clf.fit(X_train, convert_labels(y_train))
    predicted = clf.predict(X_test)
    return predicted


def train_nnrf(train, test, label, oversample="none", to_normalize=False, learning_rate=0.1, reg_factor=0, **kwargs):
    """Train an NNRF model and make predictions.

    :param train: DataFrame of the training set
    :type train: pandas.DataFrame
    :param test: DataFrame of the testing set
    :type test: pandas.DataFrame
    :param label: name of the target column for supervised learning
    :type label: str
    :param oversample: oversampling algorithm to be applied ("none", "smote", "adasyn", "borderline")
    :type oversample: str, optional
    :param to_normalize: normalize features
    :type to_normalize: bool, optional
    :param learning_rate: learning rate of optimizer
    :type learning_rate: float, optional
    :param reg_factor: L2 regularization factor
    :type reg_factor: float, optional
    :returns: array of class predictions
    :rtype: np.ndarray[int]
    """
    # Oversampling the training set
    train = oversample_data(train, label, oversample)

    o = ml.optimizer.Adam(alpha=learning_rate)
    r = ml.regularizer.L2(c=reg_factor)

    clf = NNRF(random_state=42, loss='cross-entropy', optimizer=o, regularize=r, **kwargs)
    X_train, y_train = df_to_nparray(train, label)
    X_test, y_test = df_to_nparray(test, label)

    # normalize features
    if to_normalize:
        X_train = normalize(X_train)
        X_test = normalize(X_test)

    clf.fit(X_train, convert_labels(y_train))
    predicted = clf.predict(X_test)
    return predicted


def train_dnn(train, test, label, features, layers, oversample="none"):
    """Train a dense neural network model and make predictions.

    :param train: DataFrame of the training set
    :type train: pandas.DataFrame
    :param test: DataFrame of the testing set
    :type test: pandas.DataFrame
    :param label: name of the target column for supervised learning
    :type label: str
    :param features: features to include in training
    :type features: list[str]
    :param layers: number of nodes per hidden layer in the neural network
    :type layers: list[int]
    :param oversample: oversampling algorithm to be applied ("none", "smote", "adasyn", "borderline")
    :type oversample: str, optional
    :returns: array of class predictions
    :rtype: np.ndarray[int]

    .. todo:: Build custom evaluation functions to get the model predictions with Tensorflow.
    .. todo:: This only supports binary classification.
    """
    # Generate feauture columns
    all_inputs = []
    for col in features:
        all_inputs.append(tf.keras.Input(shape=(1,), name=col))

    X_train, X_val, y_train, y_val = train_test_split(
        train.drop(label, axis=1),
        train[[label]],
        test_size=0.2,
        random_state=42,
        stratify=train[label]
    )

    train = pd.merge(X_train, y_train, left_index=True, right_index=True)
    val = pd.merge(X_val, y_val, left_index=True, right_index=True)

    # Oversampling the training set
    train = oversample_data(train, label, oversample)

    # Generating a tensorflow dataset
    train_ds = df_to_dataset(train, label)
    val_ds = df_to_dataset(val, label)
    test_ds = df_to_dataset(test, label)

    # Building the model
    all_features = tf.keras.layers.concatenate(all_inputs)
    x = tf.keras.layers.Dense(layers[0], activation="relu")(all_features)
    for nodes in layers:
        x = tf.keras.layers.Dense(nodes, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(all_inputs, output)

    # Compiling the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy',
                           tf.keras.metrics.TruePositives(),
                           tf.keras.metrics.TrueNegatives(),
                           tf.keras.metrics.FalsePositives(),
                           tf.keras.metrics.FalseNegatives()
                           ]
                  )

    # Training the Model
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model.fit(train_ds, epochs=30, callbacks=[es], validation_data=val_ds, verbose=1)

    # Getting predictions
    output = np.asarray([x[0] for x in model.predict(test_ds)])
    predicted = np.where(output >= 0.5, 1, 0)

    return predicted


def train_kfold(train_set, label, num_fold, train_func, **kwargs):
    """Validate a model with stratified k-fold cross validation.

    :param train_set: DataFrame of the training set
    :type train_set: pandas.DataFrame
    :param label: name of the target column for supervised learning
    :type label: str
    :param num_fold: number of folds
    :type num_fold: int
    :param train_func: training function of the model being validated
    :type train_func: Callable[..., (float,float,float)]
    :param **kwargs: other keyword arguments for the training function
    :returns: dictionary of metrics.
    :rtype: dict['ACCURACY'|'SENSITIVITY'|'SPECIFICITY' | 'KAPPA']['ALL'|'MEAN'|'STD']

    ..todo:: This only supports binary classification.
    """

    # Arrays for metrics
    acc_per_fold = []
    sens_per_fold = []
    spec_per_fold = []
    kappa_per_fold = []

    # Build K folds
    kfold = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=42)
    fold_no = 1
    for train_idx, val_idx in kfold.split(train_set.drop(label, axis=1), train_set[[label]]):
        train = train_set.iloc[train_idx]
        test = train_set.iloc[val_idx]

        print(f"Starting fold {fold_no}...")
        # Train model
        predicted = train_func(train, test, label, **kwargs)
        accuracy, sensitivity, specificity, kappa = get_metrics(predicted, convert_labels(test[label]))

        acc_per_fold.append(accuracy)
        sens_per_fold.append(sensitivity)
        spec_per_fold.append(specificity)
        kappa_per_fold.append(kappa)

        print(f"Fold {fold_no} completed.\n")
        fold_no += 1

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
        },
        'KAPPA': {
            'ALL': kappa_per_fold,
            'MEAN': np.mean(kappa_per_fold),
            'STDEV': np.std(kappa_per_fold)
        }
    }
    return metrics
