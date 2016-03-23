# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 23 March 2016
# Montana State University - Optical Remote Sensing Lab


import argparse
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier

sys.path.append("..")
from common import FileIO
from common.Constants import *


def main(date, keywords=[]):
    """
    Runs linear regression (classification) between the herbicide 
    resistance classes based on all wavelengths. The weights
    associated with each wavelength are then plotted, allowing
    the user to see the contribution to classification by each
    wavelength.

    :param date: (string) Data collection date YYYY_MMDD
    :param keywords: (list of strings) Data filename keywords

    :return: (None)
    """
    
    # Get the data files we will be looking at
    dataPath = DATA_DIRECTORIES[date]
    dataFilenames = FileIO.getDatafileNames(dataPath, keywords)

    # Train the classifier on the loaded data
    clf = SGDClassifier()
    clf.fit(X, y)

    # Plot the feature weights to visualize feature contributions
    featureWeights = clf.coef_



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run linear regression on the wavelengths')
    parser.add_argument('date', type=str, nargs=1,
                         help='Data collection date YYYY_MMDD')
    parser.add_argument('-k', '--keywords', default=[], type=str, nargs='*',
                         help="Filename keywords to include")

    args = parser.parse_args()
    main(args.date[0], args.keywords)
