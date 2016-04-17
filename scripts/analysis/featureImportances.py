# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 16 April 2016
# Montana State University - Optical Remote Sensing Lab


"""
Using a random forest classifier, determine the importances of the features
used in training, and display these features for decisions 

Usage:

    python featureImportances.py [date]
"""

import os
import sys
sys.path.append("..")
import argparse
import multiprocessing
import numpy as np

from common import FileIO
from common.Constants import *

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


def main(date):
    """
    Trains a random forest and extracts the feature importances.

    :param date: Date the training and testing data was collected (YYYY_MMDD)

    :return: (None)
    """

    # Load the training data into memory
    trainX, trainY = FileIO.loadTrainingData(date)
    trainX = np.nan_to_num(trainX)

    # Train the random forest on the training data
    numCores = multiprocessing.cpu_count()
    forest = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=numCores)
    forest.fit(trainX, trainY)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(trainX.shape[1]), importances[indices],
           color="r", align="center")
    plt.xticks(range(trainX.shape[1]), indices)
    plt.xlim([-1, trainX.shape[1]])
    plt.show()


if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Determine Feature Importances')
    parser.add_argument('date', type=str, nargs=1,
                         help='Data collection date YYYY_MMDD')
    args = parser.parse_args()

    main(args.date[0])