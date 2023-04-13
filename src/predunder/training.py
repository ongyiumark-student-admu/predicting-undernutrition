# Importing libraries
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from xgboost import XGBClassifier
from nnrf import NNRF
from nnrf import ml

from predunder.functions import (convert_labels, df_to_nparray, get_metrics, oversample_data)


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
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_train = X_train - X_mean / X_std
        X_test = X_train - X_mean / X_std

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


def train_dnn(train, test, label, layers, epochs=1, oversample="none"):
    """Train a dense neural network model and make predictions.

    :param train: DataFrame of the training set
    :type train: pandas.DataFrame
    :param test: DataFrame of the testing set
    :type test: pandas.DataFrame
    :param label: name of the target column for supervised learning
    :type label: str
    :param layers: number of nodes per hidden layer in the neural network
    :type layers: list[int]
    :param oversample: oversampling algorithm to be applied ("none", "smote", "adasyn", "borderline")
    :type oversample: str, optional
    :returns: array of class predictions
    :rtype: np.ndarray[int]

    .. todo:: Build custom evaluation functions to get the model predictions with Tensorflow.
    .. todo:: This only supports binary classification.
    """
    # Oversampling the training set
    train = oversample_data(train, label, oversample)

    labels = convert_labels(train[label])
    features = train.drop(label, axis=1).to_dict(orient='list')
    predict_features = test.drop(label, axis=1).to_dict(orient='list')

    INTEGER_VARIABLES = [
        'IDD_SCORE',
        'AGE',
        'HHID_count',
        'HDD_SCORE'
    ]
    BOOLEAN_VARIABLES = [
        ('CHILD_SEX', 2),
        ('BEN_4PS', 2),
        ('AREA_TYPE', 2),
    ]
    ONE_HOT_VARIABLES = [
        ('FOOD_INSECURITY', 5)
    ]
    FLOAT_VARIABLES = [
        'HH_AGE',
        'FOOD_EXPENSE_WEEKLY',
        'NON-FOOD_EXPENSE_WEEKLY',
        'FOOD_EXPENSE_WEEKLY_pc',
        'NON-FOOD_EXPENSE_WEEKLY_pc'
    ]

    def define_preprocessing_model():
        inputs = {
            **{int_var: tf.keras.Input(shape=(), dtype='int64') for int_var in INTEGER_VARIABLES},
            **{bool_var: tf.keras.Input(shape=(), dtype='bool') for bool_var, dim in BOOLEAN_VARIABLES},
            **{one_hot_var: tf.keras.Input(shape=(), dtype='int64') for one_hot_var, dim in ONE_HOT_VARIABLES},
            **{float_var: tf.keras.Input(shape=(), dtype='float64') for float_var in FLOAT_VARIABLES}
        }

        one_hot_outputs = {one_hot_var: tf.keras.layers.CategoryEncoding(dim, output_mode='one_hot')(inputs[one_hot_var])
                           for one_hot_var, dim in ONE_HOT_VARIABLES}
        bool_outputs = {bool_var: tf.keras.layers.Normalization(axis=None, mean=0, variance=1)(inputs[bool_var])
                        for bool_var, dim in BOOLEAN_VARIABLES}
        integer_outputs = {int_var: tf.keras.layers.Normalization(axis=None,
                           mean=np.mean(features[int_var]),
                           variance=np.var(features[int_var]))(inputs[int_var])
                           for int_var in INTEGER_VARIABLES}
        float_outputs = {float_var: tf.keras.layers.Normalization(axis=None,
                         mean=np.mean(features[float_var]),
                         variance=np.var(features[float_var]))(inputs[float_var])
                         for float_var in FLOAT_VARIABLES}
        outputs = {
            **one_hot_outputs,
            **bool_outputs,
            **integer_outputs,
            **float_outputs
        }
        preprocessing_model = tf.keras.Model(inputs, outputs)
        return preprocessing_model

    def define_training_model(layers):
        inputs = {
            **{int_var: tf.keras.Input(shape=(), dtype='float64') for int_var in INTEGER_VARIABLES},
            **{bool_var: tf.keras.Input(shape=(), dtype='float64') for bool_var, dim in BOOLEAN_VARIABLES},
            **{one_hot_var: tf.keras.Input(shape=(dim, ), dtype='float64') for one_hot_var, dim in ONE_HOT_VARIABLES},
            **{float_var: tf.keras.Input(shape=(), dtype='float64') for float_var in FLOAT_VARIABLES}
        }
        outputs = tf.keras.layers.Concatenate()([
            *[inputs[one_hot_var] for one_hot_var, dim in ONE_HOT_VARIABLES],
            *[tf.expand_dims(inputs[int_var], -1) for int_var in INTEGER_VARIABLES],
            *[tf.expand_dims(inputs[bool_var], -1) for bool_var, dim in BOOLEAN_VARIABLES],
            *[tf.expand_dims(inputs[float_var], -1) for float_var in FLOAT_VARIABLES]
        ])
        for layer in layers:
            outputs = tf.keras.layers.Dense(layer, activation='relu')(outputs)
        outputs = tf.keras.layers.Dense(1)(outputs)
        training_model = tf.keras.Model(inputs, outputs)
        return training_model

    preprocessing_model = define_preprocessing_model()
    training_model = define_training_model(layers)

    # Apply the preprocessing in tf.data.Dataset.map.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(1)
    dataset = dataset.map(lambda x, y: (preprocessing_model(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Compile the model
    training_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy',
                 tf.keras.metrics.TruePositives(),
                 tf.keras.metrics.TrueNegatives(),
                 tf.keras.metrics.FalsePositives(),
                 tf.keras.metrics.FalseNegatives()
                 ]
    )

    # Fit the model
    training_model.fit(dataset, epochs=epochs)

    # Get predictions
    inputs = preprocessing_model.input
    outputs = training_model(preprocessing_model(inputs))
    inference_model = tf.keras.Model(inputs, outputs)

    predict_dataset = tf.data.Dataset.from_tensor_slices(predict_features).batch(1)
    predicted = np.array(list(map(lambda x: int(x > 0.5), inference_model.predict(predict_dataset).reshape(-1))))

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
