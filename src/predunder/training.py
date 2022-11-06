# Importing libraries
import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier

from predunder.functions import (convert_labels, df_to_nparray, df_to_dataset, get_metrics,
                                 image_to_dataset, oversample_data)


def train_random_forest(train, test, label, oversample="none"):
    """Train a random forest model and make predictions.

    :param train: DataFrame of the training set
    :type train: pandas.DataFrame
    :param test: DataFrame of the testing set
    :type test: pandas.DataFrame
    :param label: name of the target column for supervised learning
    :type label: str
    :param oversample: oversampling algorithm to be applied ("none", "smote", "adasyn", "borderline")
    :type oversample: str, optional
    :returns: array of class predictions
    :rtype: np.ndarray[int]
    """
    # Oversampling the training set
    train = oversample_data(train, label, oversample)

    clf = RandomForestClassifier(max_depth=2, random_state=42)
    X_train, y_train = df_to_nparray(train, label)
    X_test, y_test = df_to_nparray(test, label)

    clf.fit(X_train, convert_labels(y_train))
    predicted = clf.predict(X_test)
    return predicted


def train_xgboost(train, test, label, oversample="none"):
    """Train an XGBoost model and make predictions.

    :param train: DataFrame of the training set
    :type train: pandas.DataFrame
    :param test: DataFrame of the testing set
    :type test: pandas.DataFrame
    :param label: name of the target column for supervised learning
    :type label: str
    :param oversample: oversampling algorithm to be applied ("none", "smote", "adasyn", "borderline")
    :type oversample: str, optional
    :returns: array of class predictions
    :rtype: np.ndarray[int]
    """
    # Oversampling the training set
    train = oversample_data(train, label, oversample)

    clf = RandomForestClassifier(max_depth=2, random_state=42)
    X_train, y_train = df_to_nparray(train, label)
    X_test, y_test = df_to_nparray(test, label)

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


# def train_naive_hive_sp(train, test, label, num_hive, train_network, **kwargs):
#     """
#         Trains a random hive by naively training neural networks of the same architecture on the whole training set.

#         :param train: pandas dataframe of the training set
#         :param test: pandas dataframe of the testing set
#         :param label: name of the target column for supervised learning
#         :param num_hive: number of networks in the hive
#         :param train_network: training function for each network
#         :param **kwargs: other keyword arguments for the training function
#         :return predicted: numpy array of class predictions
#     """

#     # Training networks
#     ballots = []
#     ballot_weights = []
#     for x in range(num_hive):
#         print(f"Training network {x+1}...")
#         preds = train_network(train, test, label, **kwargs)
#         sensitivity, specificity = get_metrics(preds, np.where(test[label].values == 'INCREASED RISK', 1, 0))[1:]
#         weights = [sensitivity if p == 0 else specificity for p in preds]
#         ballots.append(preds)
#         ballot_weights.append(weights)
#         print(f"Network {x+1} completed.")

#     # Counting votes with smart voting
#     predicted = []
#     for p, w in zip(zip(*ballots), zip(*ballot_weights)):
#         cnt0, cnt1 = (0, 0)
#         for vote, weight in zip(p, w):
#             if vote == 0:
#                 cnt0 += 1-(weight <= 0.2)
#             else:
#                 cnt1 += 1-(weight <= 0.2)
#         predicted.append(1 if cnt1 >= cnt0 else 0)

#     return np.asarray(predicted)


def train_naive_hive(train, test, label, num_hive, train_network, **kwargs):
    """Train a random hive with neural networks of the same architecture on the whole training set.

    :param train: DataFrame of the training set
    :type train: pandas.DataFrame
    :param test: DataFrame of the testing set
    :type test: pandas.DataFrame
    :param label: name of the target column for supervised learning
    :type label: str
    :param num_hive: number of networks in the hive
    :type num_hive: int
    :param train_network: training function for each network
    :type: Callable[.., numpy.ndarray[int]]
    :param **kwargs: other keyword arguments for the training function
    :returns: array of class predictions
    :rtype: numpy.ndarray[int]

    .. todo:: Implement option for smarter voting.
    """

    # Training networks
    ballots = []
    for x in range(num_hive):
        print(f"Training network {x+1}...")
        preds = train_network(train, test, label, **kwargs)
        ballots.append(preds)
        print(f"Network {x+1} completed.\n")

    # Counting votes
    predicted = []
    for p in zip(*ballots):
        votes = sum(p)
        predicted.append(1 if 2*votes >= num_hive else 0)

    return np.asarray(predicted)


def train_images(train, test, label, convert_func, img_size, tmp_dir, oversample="none", base_model="mobilenetv2", learning_rate=0.0001):
    """Converts tabular data into images and train a classifier with transfer learning.

    :param train: DataFrame of the training set
    :type train: pandas.DataFrame
    :param test: DataFrame of the testing set
    :type test: pandas.DataFrame
    :param label: name of the target column for supervised learning
    :type label: str
    :param convert_func: table to image converter function
    :type convert_func: Callable[.., None]
    :param img_size: dimension of the resulting images
    :type img_size: (int, int)
    :param tmp_dir: output directory where the images will be temporarily stored
    :type tmp_dir: str
    :param base_model: base model for transfer learning from Keras_
    :type base_model: str, optional
    :param learning_rate: learning rate for the neural network
    :type learning_rate: float, optional
    :returns: array of class predictions
    :rtype: numpy.array[int]

    .. _Keras: https://keras.io/api/applications/
    .. todo:: Build custom evaluation functions to get the model predictions with Tensorflow, and implement early stopping from validation loss.
    """
    # Deleting directory if exists
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    X_train, X_val, y_train, y_val = train_test_split(
        train.drop(label, axis=1),
        train[[label]],
        test_size=0.2,
        random_state=42,
        stratify=train[label]
    )

    train = pd.merge(X_train, y_train, left_index=True, right_index=True)
    val = pd.merge(X_val, y_val, left_index=True, right_index=True)

    featuredf = train.drop([label], axis=1)
    # Oversampling the training set
    train = oversample_data(train, label, oversample)

    # Generating images
    convert_func(train, featuredf.mean(), featuredf.std(), label, img_size, os.path.join(tmp_dir, 'train'))
    convert_func(val, featuredf.mean(), featuredf.std(), label, img_size, os.path.join(tmp_dir, 'val'))
    convert_func(test, featuredf.mean(), featuredf.std(), label, img_size, os.path.join(tmp_dir, 'test'))

    # Generating tensorflow dataset
    train_ds = image_to_dataset(os.path.join(tmp_dir, 'train'), img_size)
    val_ds = image_to_dataset(os.path.join(tmp_dir, 'val'), img_size)
    test_ds = image_to_dataset(os.path.join(tmp_dir, 'test'), img_size)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # Building the model
    img_shape = img_size+(3,)
    if base_model == 'xception':
        bm = tf.keras.applications.Xception(input_shape=img_shape, include_top=False, weights='imagenet')
        preprocess = tf.keras.applications.xception.preprocess_input
    elif base_model == 'resnet50v2':
        bm = tf.keras.applications.ResNet50V2(input_shape=img_shape, include_top=False, weights='imagenet')
        preprocess = tf.keras.applications.resnet_v2.preprocess_input
    elif base_model == 'mobilenetv2':
        bm = tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif base_model == 'efficientnetv2s':
        bm = tf.keras.applications.EfficientNetV2S(input_shape=img_shape, include_top=False, weights='imagenet')
        preprocess = tf.keras.applications.efficientnet_v2.preprocess_input

    inputs = tf.keras.Input(shape=img_shape)
    x = preprocess(inputs)
    x = bm(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, output)

    # Compiling the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy',
                           tf.keras.metrics.TruePositives(),
                           tf.keras.metrics.TrueNegatives(),
                           tf.keras.metrics.FalsePositives(),
                           tf.keras.metrics.FalseNegatives()],
                  )

    # Training the Model
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model.fit(train_ds, epochs=20, callbacks=[es], validation_data=val_ds, verbose=1)

    # Getting predictions
    output = np.asarray([x[0] for x in model.predict(test_ds)])
    predicted = np.where(output >= 0.5, 1, 0)

    # Deleting temporary image folder
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

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
    :type train_func: Callable[.., (flaot,flaot,flaot)]
    :param **kwargs: other keyword arguments for the training function
    :returns: dictionary of metrics.
    :rtype: dict['ACCURACY'|'SENSITIVITY'|'SPECIFICITY']['ALL'|'MEAN'|'STD']

    ..todo:: This only supports binary classification.
    """

    # Arrays for metrics
    acc_per_fold = []
    sens_per_fold = []
    spec_per_fold = []

    # Build K folds
    kfold = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=42)
    fold_no = 1
    for train_idx, val_idx in kfold.split(train_set.drop(label, axis=1), train_set[[label]]):
        train = train_set.iloc[train_idx]
        test = train_set.iloc[val_idx]

        print(f"Starting fold {fold_no}...")
        # Train model
        predicted = train_func(train, test, label, **kwargs)
        accuracy, sensitivity, specificity = get_metrics(predicted, convert_labels(test[label]))

        acc_per_fold.append(accuracy)
        sens_per_fold.append(sensitivity)
        spec_per_fold.append(specificity)

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
        }
    }
    return metrics
