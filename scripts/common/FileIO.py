# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 27 October 2015
# Montana State University - Optical Remote Sensing Lab

"""
This file contains functions that relate to reading and writing
to disk. Examples include reading and writing the training and testing
data, machine learning models, etc. 
"""

import os
import cPickle
import numpy as np


def loadCSV(filepath):
    """
    Loads in a matrix of data from a CSV file. This function 
    simply returns the data contained in the file - nothing more.

    :param filepath: (string) Path of CSV file to load

    :return: (np.array) 2-d numpy array of floats
    """

    return np.loadtxt(filepath, delimiter=',')


def getDatafileNames(directory, keywords=[]):
    """
    Returns the list of files within a given directory which
    are CSV files (useful data files).

    :param directory: (string) Directory to search for datafiles
    :param keywords: (list of strings) keywords to be included in all file names

    :return: (list of strings) CSV filenames within the directory
    """

    filenames = []

    if keywords == []:
        for name in os.listdir(directory):
            name = name.lower()
            if name.endswith('.csv'):
                filenames.append(name)
    else:
        for name in os.listdir(directory):
            name = name.lower()
            if name.endswith('.csv'):

                containsKeyword = False
                for kw in keywords:
                    containsKeyword |= (kw in name)

                if containsKeyword:
                    filenames.append(name)
    return filenames


def loadTrainingData():
    """
    Loads the training data from disk. Note, because of the potential
    for very large amounts of training data, the training data is returned
    as a numpy memmap.

    :param path: (string) location of the training data

    :return: tuple containing the training data (row ordered feature vectors),
             as well as the target labels (X, y). X has type np.memmap, and 
             y has type np.array
    """
    
    X = None
    y = None

    return X, y


def loadTestingData():
    """
    Loads the testing data from disk. Note, because of the potential
    for very large amounts of testing data, the testing data is returned
    as a numpy memmap.

    :param path: (string) location of the testing data

    :return: tuple containing the testing data (row ordered feature vectors),
             as well as the target labels (X, y). X has type np.memmap, and 
             y has type np.array
    """
    
    X = None
    y = None

    return X, y


def saveModel(model, path):
    """
    Saves a trained machine learning model to the specified
    path. This method must be called before the models can
    be used for application purposes.

    :param model: (sklearn model) trained machine learning model
    :param path: (string) destination path for the model

    :return: (None)
    """

    with open(path, 'wb') as fileHandle:
        cPickle.dump(model, fileHandle)
        fileHandle.close()


def loadModel(path):
    """
    Loads a trained machine learning model from disk for use.

    :param path: (string) path for the model fileHandle

    :return: (sklearn model) trained machine learning model
    """

    with open(path, 'rb') as fileHandle:
        model = cPickle.load(fileHandle)
        fileHandle.close()

    return model

# Other possible methods:
# - saving training/testing data
# - deleting training/testing data
# - saving PCA models
# - loading metadata saved with the training/testing data


