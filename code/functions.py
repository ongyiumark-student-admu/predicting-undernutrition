# Importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# Creating aliases
PandasDataFrame = pd.DataFrame
TensorflowDataset = tf.data.Dataset
NumpyArrayPair = tuple[np.ndarray, np.ndarray]

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