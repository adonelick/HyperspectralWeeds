# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 27 October 2015
# Montana State University - Optical Remote Sensing Lab

"""
This script is used to train machine learning models on the gathered
training data. The trained model is then saved to 

Very basic tests are run on the model to check the accuracy
of training. For more complete testing, please see the 'analysis' scripts.

Usage:
    
    To train a machine learning model, call this script with the following
    command on the command line:

    python train.py [modelType]

    The 'modelType' parameter specifies which model to train. The choices for
    the different models are defined at the top of this file.
"""

import sys
sys.path.append("..")
import argparse
from common import FileIO

# Import the required machine learning models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import NeuralNetworkClassifier


# Constants which define the machine learning models' abbreviations
SVM = "svm"
K_NEAREST_NEIGHBORS = "knn"
DECISION_TREE = "dt"
RANDOM_FOREST = "rf"
EXTREMELY_RANDOMIZED_TREES = "ert"
ADABOOST = "ab"
GRADIENT_BOOSTING = "gb"
NEURAL_NETWORK = "nn"
ALL = "all"

MODELS = {
    SVM : ("Support Vector Machine", SVC),
    K_NEAREST_NEIGHBORS : ("K-Nearest Neighbors", KNeighborsClassifier),
    DECISION_TREE : ("Decision Tree", DecisionTreeClassifier),
    RANDOM_FOREST : ("Random Forest", RandomForestClassifier),
    EXTREMELY_RANDOMIZED_TREES : ("Extra Random Trees", ExtraTreesClassifier),
    ADABOOST : ("Adaboost", AdaBoostClassifier),
    GRADIENT_BOOSTING : ("Gradient Boost", GradientBoostingClassifier),
    NEURAL_NETWORK : ("Neural Network", NeuralNetworkClassifier)
}


# Hyperparameters used in training the machine learning models
HYPERPARAMETERS = {
    SVM : {},
    K_NEAREST_NEIGHBORS : {},
    DECISION_TREE : {},
    RANDOM_FOREST : {},
    EXTREMELY_RANDOMIZED_TREES : {},
    ADABOOST : {},
    GRADIENT_BOOSTING : {},
    NEURAL_NETWORK : {}
}


def main(modelType):
    """
    Runs the training script. Trains the specified model type, saves the 
    model to a prefined location (specified in the Constants file), and 
    runs basic accuracy tests on the trained model.

    :param modelType: (string) type of machine learning model to train

    :return: (None)
    """
    
    # Make sure that the model is a valid choice
    if (not modelType in MODELS.keys()) and (modelType != ALL):
        print "Invalid model type:", modelType
        return

    # Allow for training more than one model at a time
    if modelType == ALL:
        modelsToTrain = MODELS.keys()
    else:
        modelsToTrain = [modelType]

    # Load the training and testing data into memory
    trainX, trainY = FileIO.loadTrainingData()
    testX, testY = FileIO.loadTestingData()

    for modelType in modelsToTrain:

        # Train the desired ML model
        name, clf = MODELS[modelType]
        print "Training the ", name
        clf.fit(trainX, trainY)

        # Perform some very basic accuracy testing

        # Save the model to disk




if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Train machine learning models.')
    parser.add_argument('modelType', metavar='modelType', type=str, 
                                     help='ML model identifier')
    args = parser.parse_args()
    main(args.modelType)
