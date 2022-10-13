# Importing libraries
import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from typing import Callable
import numpy.typing as npt
from predunder.functions import df_to_dataset, image_to_dataset, oversample_data, get_metrics

# Defining Aliases
PandasDataFrame = pd.core.frame.DataFrame


def train_dnn(train: PandasDataFrame, test: PandasDataFrame, label: str, features: list[str], layers: list[int],
              oversample: str = "none") -> npt.NDArray[np.int64]:
    """
        Trains a dense neural network model with 'train' and evaluates the model on 'test'.

        :param train: pandas dataframe of the training set
        :param test: pandas dataframe of the testing set
        :param label: name of the target column for supervised learning
        :param features: list of features to include in training
        :param layers: list of number of nodes per layer
        :param oversample: oversampling algorithm to be applied
        :return predicted: numpy array of class predictions
    """
    # Generate feauture columns
    all_inputs = []
    for col in features:
        all_inputs.append(tf.keras.Input(shape=(1,), name=col))

    # Oversampling the training set
    train = oversample_data(train, label, oversample)

    # Generating a tensorflow dataset
    train_ds = df_to_dataset(train, label)
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
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.fit(train_ds, epochs=20, callbacks=[es], verbose=1)

    # Getting predictions
    output = np.asarray([x[0] for x in model.predict(test_ds)])
    predicted = np.where(output >= 0.5, 1, 0)

    return predicted


def train_naive_hive(train: PandasDataFrame, test: PandasDataFrame, label: str, num_hive: int, train_network: Callable, **kwargs) -> npt.NDArray[np.int64]:
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
    ballot_weights = []
    for x in range(num_hive):
        print(f"Training network {x+1}...")
        preds = train_network(train, test, label, **kwargs)
        sensitivity, specificity = get_metrics(preds, np.where(test[label].values == 'INCREASED RISK', 1, 0))[1:]
        weights = [sensitivity if p == 0 else specificity for p in preds]
        ballots.append(preds)
        ballot_weights.append(weights)
        print(f"Network {x+1} completed.")

    # Counting votes with smart voting
    predicted = []
    for p, w in zip(zip(*ballots), zip(*ballot_weights)):
        cnt0, cnt1 = (0, 0)
        for vote, weight in zip(p, w):
            if vote == 0:
                cnt0 += 1-(weight <= 0.2)
            else:
                cnt1 += 1-(weight <= 0.2)
        predicted.append(1 if cnt1 >= cnt0 else 0)

    return np.asarray(predicted)


def train_images(train: PandasDataFrame, test: PandasDataFrame, label: str, convert_func: Callable, img_size: tuple[int, int],
                 tmpdir: str, oversample: str = "none", base_model: str = "mobilenetv2", learning_rate: float = 0.0001) -> npt.NDArray[np.int64]:
    """
        Converts tabular data into images and trains with transfer learning.

        :param train: pandas dataframe of the training set
        :param test: pandas dataframe of the testing set
        :param label: name of the target column for supervised learning
        :param convert_func: table to image converter function
        :param img_size: dimensions of the resulting images
        :param tmpdir: directory where the images will be temporarily stored
        :param base_model: base model for transfer learning
        :param learning_rate: learning rate for the neural network
        :return predicted: numpy array of class predictions
    """
    # Deleting directory if exists
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)

    featuredf = train.drop([label], axis=1)
    # Oversampling the training set
    train = oversample_data(train, label, oversample)

    # Generating images
    convert_func(train, featuredf.mean(), featuredf.std(), label, img_size, os.path.join(tmpdir, 'train'))
    convert_func(test, featuredf.mean(), featuredf.std(), label, img_size, os.path.join(tmpdir, 'test'))

    # Generating tensorflow dataset
    train_ds = image_to_dataset(os.path.join(tmpdir, 'train'), img_size)
    test_ds = image_to_dataset(os.path.join(tmpdir, 'test'), img_size)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
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
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.fit(train_ds, epochs=20, callbacks=[es], verbose=1)

    # Getting predictions
    output = np.asarray([x[0] for x in model.predict(test_ds)])
    predicted = np.where(output >= 0.5, 1, 0)

    # Deleting temporary image folder
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)

    return predicted


def train_kfold(train_set: PandasDataFrame, label: str, num_fold: int, train_func: Callable, **kwargs) -> dict:
    """
        Validates a model with stratified k-fold cross validation.

        :param train_set: pandas dataframe of the training set
        :param label: name of the target column for supervised learning
        :param num_fold: number of folds
        :param train_func: training function of the model being validated
        :param to_smote: flag for applying oversampling with SMOTE
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
