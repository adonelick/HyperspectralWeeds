# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 23 March 2016
# Montana State University - Optical Remote Sensing Lab

"""
This file contains functions that relate to generating 
training and testing data from individual datafiles.
"""

import os
import cPickle
import numpy as np
from Constants import *


def separateTrainTest(dataPath, filenames, byLeaf=True, saveProportion=0.5):
    """
    Consolidates a set of data files (one file per specific leaf)
    into two numpy arrays: one of training data, another for testing
    data. Vectors of labels for this data are also generated.

    :param dataPath: (string) Root directory for the data files
    :param filenames: (list of strings) Data files to be separated
    :param byLeaf: (boolean) Should we separate the train/test data
                             by leaf, or should we randomly separate
                             the data according to a set proportion?
    :param saveProportion: (float) Proportion of the data saved as training/testing data

    :return: (tuple of np.arrays) Training and testing data
                                  (train_X, train_y, test_X, test_y)
    """

    train_X = None
    train_y = np.array([])
    test_X = None
    test_y = np.array([])

    for name in filenames:

        filePath = os.path.join(dataPath, name)
        data = np.loadtxt(filePath, delimiter=',')
        try:
            samples, features = data.shape
        except:
            continue
        
        # Get the proper label for the data
        name = name.lower()
        if SUSCEPTIBLE in name:
            label = LABEL_TO_INDEX[SUSCEPTIBLE]
        elif GR_RESISTANT in name:
            label = LABEL_TO_INDEX[GR_RESISTANT]
        elif DR_RESISTANT in name:
            label = LABEL_TO_INDEX[DR_RESISTANT]
        else:
            raise Exception("Resistance class not found in filename!")

        np.random.shuffle(data)
        data = data[0:int(saveProportion*samples), :]
        samples, features = data.shape

        # Now that we have the data, we can sort it into 
        # the training or testing pile
        if byLeaf:

            if np.random.rand() < TRAIN_PROPORTION:

                if train_X == None:
                    train_X = data
                else:
                    train_X = np.append(train_X, data, axis=0)
                train_y = np.append(train_y, [label]*samples)

            else:
                
                if test_X == None:
                    test_X = data
                else:
                    test_X = np.append(test_X, data, axis=0)
                test_y = np.append(test_y, [label]*samples)

        else:
            
            trainIndex = int(TRAIN_PROPORTION * samples)

            if train_X == None:
                train_X = data[0:trainIndex]
            else:
                train_X = np.append(train_X, data[0:trainIndex], axis=0)
            train_y = np.append(train_y, [label]*trainIndex)


            if test_X == None:
                test_X = data[trainIndex:]
            else:
                test_X = np.append(test_X, data[trainIndex:], axis=0)
            test_y = np.append(test_y, [label]*(samples - trainIndex))

    return (train_X, train_y, test_X, test_y)



def updateTrainingData(date, X, y):
    """
    Updates a training data set with new training data.

    :param date: (string) Date in which the data was collected (YYYY_MMDD)
    :param X: (np.array) Array of training features (n_samples x n_features)
    :param y: (np.array) Array of labels for the training data (n_samples)

    :return: (None)
    """

    sampleCounts = loadSampleCounts(date)
    dataDirectory = DATA_DIRECTORIES[date+"_ML"]
    trainingDataPath = os.path.join(dataDirectory, TRAINING_DATA_PATH)

    trainingSamples = sampleCounts["training"]
    samples, features = X.shape

    # Open the file for appending
    trainingData = np.memmap(trainingDataPath, 
                             mode='r+',
                             dtype=np.float32,
                             shape=(trainingSamples, features+1))

    trainingData = trainingData.copy()
    trainingData.resize(trainingSamples + samples, features+1)
    
    trainingData[trainingSamples:, 0] = y
    trainingData[trainingSamples:, 1:] = X

    # Update the sample counts file
    for index in y:
        labelString = INDEX_TO_LABEL[index]
        sampleCounts[labelString+"_training"] += 1
    
    sampleCounts["training"] += len(y)
    updateSampleCounts(date, sampleCounts)

    del trainingData
    

def updateTestingData(date, X, y):
    """
    Updates a testing data set with new testing data.

    :param date: (string) Date in which the data was collected (YYYY_MMDD)
    :param X: (np.array) Array of testing features (n_samples x n_features)
    :param y: (np.array) Array of labels for the testing data (n_samples)

    :return: (None)
    """

    sampleCounts = loadSampleCounts(date)
    dataDirectory = DATA_DIRECTORIES[date+"_ML"]
    testingDataPath = os.path.join(dataDirectory, TESTING_DATA_PATH)

    testingSamples = sampleCounts["testing"]
    samples, features = X.shape

    # Open the file for appending
    testingData = np.memmap(testingDataPath, 
                            mode='r+',
                            dtype=np.float32,
                            shape=(testingSamples, features+1))

    testingData = testingData.copy()
    testingData.resize(testingSamples + samples, features+1)
    
    testingData[testingSamples:, 0] = y
    testingData[testingSamples:, 1:] = X

    # Update the sample counts file
    for index in y:
        labelString = INDEX_TO_LABEL[index]
        sampleCounts[labelString+"_testing"] += 1
    
    sampleCounts["testing"] += len(y)
    updateSampleCounts(date, sampleCounts)

    del testingData


def loadSampleCounts(date):
    """
    Determine the number of training and testing samples which have already
    been saved to disk. This function looks into the pickle file
    which stores this information.

    :param date: (string) Date string (YYYY_MMDD) for when the data was collected

    :return: (dictionary) dictionary of the number of samples in each class
                          (for both training and testing), as well as the
                          total number of training and testing samples.
    """

    dataDirectory = DATA_DIRECTORIES[date + "_ML"]
    sampleCountsPath = os.path.join(dataDirectory, SAMPLE_COUNTS_PATH)

    if not os.path.exists(sampleCountsPath):
        # If the sample counts file does not exist, then 
        # create on with all the class names currently in the 
        # constants file.

        sampleCounts = {}
        for className in CLASSES:
            sampleCounts[className+"_training"] = 0
            sampleCounts[className+"_testing"] = 0

        sampleCounts["training"] = 0
        sampleCounts["testing"] = 0

    else:

        sampleCounts = cPickle.load(open(sampleCountsPath, 'rb'))

        # Verify that we have not added additional classes to the possible
        # set of classes since sampleCounts was last updated. If so, add
        # the new classes to the sampleCounts file 
        expectedKeys = ["training", "testing"]
        for className in CLASSES:
            expectedKeys.append(className+"_training")
            expectedKeys.append(className+"_testing")

        actualKeys = sampleCounts.keys()
        if actualKeys != expectedKeys:

            for key in expectedKeys:
                if key not in actualKeys:
                    sampleCounts[key] = 0

    return sampleCounts


def updateSampleCounts(date, sampleCounts):
    """
    Updates the number of training and testing samples recorded
    in the pickle file which stores such things.

    :param date: (string) Date string (YYYY_MMDD) for when the data was collected
    :param sampleCounts: (dictionary) dictionary of the number of samples
                                      in each class (for both training 
                                      and testing), as well as the total 
                                      number of training and testing samples.

    :return: (None)
    """

    dataDirectory = DATA_DIRECTORIES[date + "_ML"]
    sampleCountsPath = os.path.join(dataDirectory, SAMPLE_COUNTS_PATH)
    cPickle.dump(sampleCounts, open(sampleCountsPath, 'wb'))

