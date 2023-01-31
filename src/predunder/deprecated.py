# Importing libraries
import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

from predunder.functions import (convert_labels, oversample_data)


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
