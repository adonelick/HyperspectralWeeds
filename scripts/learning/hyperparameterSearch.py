# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 11 April 2016
# Montana State University - Optical Remote Sensing Lab

"""
This script is used to train machine learning models on the gathered
training data. The trained model is then saved to a directory, from
which it can be read and applied to a datacube in Spectronon. In training
the model, the optimal hyperparameters are determined and printed along 
with the accuracies. 

Very basic tests are run on the model to check the accuracy
of training. For more complete testing, please see the 'analysis' scripts.
(These analysis scripts are not yet implemented).

Usage:

    python hyperparameterSearch.py [date] [modelType] [iterations]

    date: Date the training and testing data were collected (YYYY_MMDD)
    modelType: parameter specifies which model to train. The choices for
               the different models are defined at the top of this file.
    iterations: Number of iterations to use when searching for the optimal
                parameters. The more iterations, the finer the search
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

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from scipy.stats import uniform
from scipy.stats import expon
from scipy.stats import randint


PARAMETERS = {
        SVM : {'kernel' : ['linear', 'rbf'], 
               'C' : expon(scale=100),
               'gamma': expon(scale=0.1),
               'class_weight' : ['auto', None]
              },
        K_NEAREST_NEIGHBORS : {'n_neighbors' :  randint(1, 20)},
        DECISION_TREE : {'criterion' : ['gini', 'entropy'],
                         'max_features' : ['auto', 'log2', None]
                        },
        RANDOM_FOREST : {'n_estimators' :  randint(10, 300),
                         'max_features' : ['auto', 'log2', None],
                         'n_jobs' : [1]
                        },
        EXTREMELY_RANDOMIZED_TREES : {'n_estimators' :  randint(10, 300),
                                      'max_features' : ['auto', 'log2', None],
                                      'n_jobs' : [1]
                                     },
        ADABOOST : {'n_estimators' : randint(10, 300)},
        GRADIENT_BOOSTING : {'loss' : ['deviance', 'exponential'],
                             'n_estimators' :  randint(50, 300)
                            }
}



def main(date, modelType, iterations):
    """
    Determines the optimal hyperparameters for a given machine learning
    model for a set of training data.

    :param date: Date the training and testing data was collected (YYYY_MMDD)
    :param modelType: (string) type of machine learning model to train
    :param iterations: (int) number of iterations for hyperparameter searching

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
        print "Training the", name

        baseClassifier = clfType()
        clf = RandomizedSearchCV(baseClassifier, param_distributions=PARAMETERS[modelType],
                                                 n_iter=iterations,
                                                 n_jobs=8)
        clf.fit(trainX, trainY)

        # Perform some very basic accuracy testing
        trainResult = clf.predict(trainX)
        testResult = clf.predict(testX)

        trainingAccuracy = accuracy_score(trainY, trainResult)
        testingAccuracy = accuracy_score(testY, testResult)
        confusionMatrix = confusion_matrix(testY, testResult)

        print "Training Accuracy:", trainingAccuracy
        print "Testing Accuracy:", testingAccuracy
        print "Confusion Matrix:"
        print confusionMatrix
        print " "
        print "Hyperparameters:"
        for param in PARAMETERS[modelType].keys():
            print param + ':', clf.best_estimator_.get_params()[param]
        print " "

        # Save the model to disk
        FileIO.saveModel(clf.best_estimator_, modelType, date)



if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Determine ML hyperparameters')
    parser.add_argument('date', type=str, nargs=1,
                         help='Data collection date YYYY_MMDD')
    parser.add_argument('modelType', type=str, nargs=1,
                         help='Type of machine learning model to train')
    parser.add_argument('iterations', type=int, nargs=1,
                         help='Numbers of iterations to search for optimal parameters')
    args = parser.parse_args()

    main(args.date[0], args.modelType[0], args.iterations[0])
