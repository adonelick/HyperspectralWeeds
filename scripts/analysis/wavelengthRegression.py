# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 29 March 2016
# Montana State University - Optical Remote Sensing Lab

"""
This script allows the user to get a sense of which wavelengths are
contributing to the difference between classes by displaying the
weight associated with each after training a linear classification
on the training data.

Usage:
    
    python wavelengthRegression.py [date]

    date: Date the training and testing data were collected (YYYY_MMDD)
"""

import argparse
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier

sys.path.append("..")
from common import FileIO
from common.Constants import *


def main(date):
    """
    Runs linear regression (classification) between the herbicide 
    resistance classes based on all wavelengths. The weights
    associated with each wavelength are then plotted, allowing
    the user to see the contribution to classification by each
    wavelength.

    :param date: (string) Data collection date YYYY_MMDD

    :return: (None)
    """
    
    # Load the training data from disk   
    X, y = FileIO.loadTrainingData(date)
    X = np.nan_to_num(X)

    # Train the classifier on the loaded data
    clf = SGDClassifier()
    clf.fit(X, y)

    # Plot the feature weights to visualize feature contributions
    featureWeights = np.fabs(clf.coef_)

    for i in xrange(3):
        plt.plot(WAVELENGTHS, featureWeights[i])
        plt.title("Linear Classifier Weights for " + RESISTANCE_STRINGS[INDEX_TO_LABEL[i]] + " vs Others")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Absolute Weight")
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run linear regression on the wavelengths')
    parser.add_argument('date', type=str, nargs=1,
                         help='Data collection date YYYY_MMDD')

    args = parser.parse_args()
    main(args.date[0])
