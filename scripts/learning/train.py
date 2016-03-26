# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 25 March 2016
# Montana State University - Optical Remote Sensing Lab

"""
This script is used to train machine learning models on the gathered
training data. The trained model is then saved to 

Very basic tests are run on the model to check the accuracy
of training. For more complete testing, please see the 'analysis' scripts.

Usage:
    
    To train a machine learning model, call this script with the following
    command on the command line:

    python train.py [date] [modelType]

    date: Date the training and testing data was collected (YYYY_MMDD)
    modelType: parameter specifies which model to train. The choices for
               the different models are defined at the top of this file.
"""

import os
import sys
sys.path.append("..")
import argparse
import numpy as np

from common import FileIO
from common.Constants import *

# Import the required machine learning models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import NeuralNetworkClassifier

from sklearn.metrics import accuracy_score


def main(date, modelType):
    """
    Runs the training script. Trains the specified model type, saves the 
    model to a prefined location (specified in the Constants file), and 
    runs basic accuracy tests on the trained model.

    :param date: Date the training and testing data was collected (YYYY_MMDD)
    :param modelType: (string) type of machine learning model to train

    :return: (None)
    """
    
    # Make sure that the model is a valid choice
    if (not (modelType in MODELS.keys())) and (modelType != ALL):
        print "Invalid model type:", modelType
        return

    # Allow for training more than one model at a time
    if modelType == ALL:
        modelsToTrain = MODELS.keys()
    else:
        modelsToTrain = [modelType]

    # Load the training and testing data into memory
    trainX, trainY = FileIO.loadTrainingData(date)
    testX, testY = FileIO.loadTestingData(date)

    trainX = np.nan_to_num(trainX)
    testX = np.nan_to_num(testX)

    for modelType in modelsToTrain:

        # Train the desired ML model
        name, clfType = MODELS[modelType]
        print "Training the ", name

        clf = clfType()
        clf.fit(trainX, trainY)

        # Perform some very basic accuracy testing

        trainResult = clf.predict(trainX)
        testResult = clf.predict(testX)

        trainingAccuracy = accuracy_score(trainY, trainResult)
        testingAccuracy = accuracy_score(testY, testResult)

        print "Training Accuracy:", trainingAccuracy
        print "Testing Accuracy:", testingAccuracy

        # Save the model to disk
        modelDirectory = MODEL_DIRECTORIES[date]
        modelPath = os.path.join(modelDirectory, name + ".model")
        # FileIO.saveModel(clf, modelPath)



if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Train machine learning models')
    parser.add_argument('date', type=str, nargs=1,
                         help='Data collection date YYYY_MMDD')
    parser.add_argument('modelType', type=str, nargs=1,
                         help='Type of machine learning model to train')
    args = parser.parse_args()

    main(args.date[0], args.modelType[0])
