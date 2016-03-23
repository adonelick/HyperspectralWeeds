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
import DataManipulation
from Constants import *


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

    :param paths: (list of strings) locations of the training data

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


def saveTrainingData(date, X, y):
    """
    Saves a given matrix of training data as a np.memmap
    in the proper location for later use in training machine
    learning models.

    :param date: (string) Date in which the data was collected (YYYY_MMDD)
    :param X: (np.array) Array of training features (n_samples x n_features)
    :param y: (np.array) Array of labels for the training data (n_samples)

    :return: (None)
    """

    sampleCounts = DataManipulation.loadSampleCounts(date)
    dataDirectory = DATA_DIRECTORIES[date+"_ML"]
    trainingDataPath = os.path.join(dataDirectory, TRAINING_DATA_PATH)

    if not os.path.exists(trainingDataPath):
        # Open the file for the first time to write
        samples, features = X.shape
        trainingData = np.memmap(trainingDataPath, 
                                 mode='w+',
                                 dtype=np.float32, 
                                 shape=(samples, features+1))

        trainingData[:, 0] = y
        trainingData[:, 1:] = X

        # Flush the data to disk and close the memmap
        del trainingData

    else:
        DataManipulation.updateTrainingData(date, X, y)
    
    # Update the sample counts file
    for index in y:
        labelString = INDEX_TO_LABEL[index]
        sampleCounts[labelString+"_training"] += 1
    
    sampleCounts["training"] += len(y)
    DataManipulation.updateSampleCounts(date, sampleCounts)


def saveTestingData(date, X, y):
    """
    Saves a given matrix of testing data as a np.memmap
    in the proper location for later use in testing
    machine learning models' performance.

    :param date: (string) Date in which the data was collected (YYYY_MMDD)
    :param X: (np.array) Array of testing features (n_samples x n_features)
    :param y: (np.array) Array of labels for the testing data (n_samples)

    :return: (None)
    """

    sampleCounts = DataManipulation.loadSampleCounts(date)
    dataDirectory = DATA_DIRECTORIES[date+"_ML"]
    testingDataPath = os.path.join(dataDirectory, TESTING_DATA_PATH)

    if not os.path.exists(testingDataPath):
        # Open the file for the first time to write
        samples, features = X.shape
        testingData = np.memmap(testingDataPath, 
                                mode='w+',
                                dtype=np.float32, 
                                shape=(samples, features+1))

        testingData[:, 0] = y
        testingData[:, 1:] = X

        # Flush the data to disk and close the memmap
        del testingData
    else:
        DataManipulation.updateTestingData(date, X, y)
    
    # Update the sample counts file
    for index in y:
        labelString = INDEX_TO_LABEL[index]
        sampleCounts[labelString+"_testing"] += 1
    
    sampleCounts["testing"] += len(y)
    DataManipulation.updateSampleCounts(date, sampleCounts)


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


