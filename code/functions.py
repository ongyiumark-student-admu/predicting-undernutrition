# Importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf

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

def train_nn(train : PandasDataFrame, test : PandasDataFrame, features : list[str], label : str, layers : list[int]) -> ModelMetrics:
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