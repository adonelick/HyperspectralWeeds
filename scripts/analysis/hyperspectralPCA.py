# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 26 March 2016
# Montana State University - Optical Remote Sensing Lab

"""
This script reduces the dimensionality of the training data to 3 dimensions,
plots the transformed data in 3d space. The idea is to bring
out separability between the resistance classes which may be 
hidden in the dimensionality of the data.

Usage:

    python hyperspectralPCA.py [date] [-s subset]

    date: Data collection date YYYY_MMDD
    subset: Transform and plot a random subset of the trainng data?

"""

import argparse
import numpy as np
import os
import sys
import mkl
sys.path.append("..")

from matplotlib import pyplot as plt

from sklearn.decomposition import IncrementalPCA

import plotly.plotly as py
import plotly.graph_objs as go

from common import FileIO
from common import Constants


NUM_SAMPLES = 1000


def main(date, takeSubset=False):
    """
    Reduces the dimensionality of the training data to 3 dimensions, 
    plots the transformed data in 3d space. The idea is to bring
    out separability between the resistance classes which may be 
    hidden in the dimensionality of the data.

    :param date: (string) Data collection date YYYY_MMDD
    :param takeSubset: (boolean) Transform and plot a random subset of
                                 the trainng data?

    :return: (None)
    """

    mkl.set_num_threads(8)

    # Load the training and testing data into memory
    trainX, trainY = FileIO.loadTrainingData(date)

    if takeSubset:
        indices = np.random.choice(range(0, len(trainY)), size=NUM_SAMPLES, replace=False)
        X = trainX[indices,:]
        y = trainY[indices]
    else:
        X = trainX
        y = trainY

    X = np.nan_to_num(X)

    # Break the data into resistance classes
    susIndex = Constants.LABEL_TO_INDEX[Constants.SUSCEPTIBLE]
    drIndex = Constants.LABEL_TO_INDEX[Constants.DR_RESISTANT]
    grIndex = Constants.LABEL_TO_INDEX[Constants.GR_RESISTANT]

    susX = X[y==susIndex, :]
    drX = X[y==drIndex, :]
    grX = X[y==grIndex, :]

    # Transform the data using PCA
    pca = IncrementalPCA(n_components=6)

    pointsSUS = pca.fit_transform(susX)
    pointsGR= pca.fit_transform(grX)
    pointsDR = pca.fit_transform(drX)

    # Plot the transformed data in 3D space
    traceSUS = go.Scatter3d(
        x=pointsSUS[:, 0],
        y=pointsSUS[:, 1],
        z=pointsSUS[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            line=dict(
                color='rgba(255, 0, 0, 0)',
                width=0.1
            ),
            opacity=0
        )
    )

    traceDR = go.Scatter3d(
        x=pointsDR[:, 0],
        y=pointsDR[:, 1],
        z=pointsDR[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            line=dict(
                color='rgba(0, 255, 0, 0)',
                width=0.1
            ),
            opacity=0
        )
    )

    traceGR = go.Scatter3d(
        x=pointsGR[:, 0],
        y=pointsGR[:, 1],
        z=pointsGR[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            line=dict(
                color='rgba(0, 0, 255, 0)',
                width=0.1
            ),
            opacity=0
        )
    )

    data = [traceSUS, traceDR, traceGR]
    fig = go.Figure(data=data)
    py.iplot(fig, filename='3D PCA Wavelength Plot')

    # Plot the principle components
    eigenSpectra = pca.components_
    print eigenSpectra.shape

    plt.subplot(3,1,1)
    plt.plot(Constants.WAVELENGTHS, eigenSpectra[0, :])
    plt.title("Principle Components 1 - 3")
    plt.subplot(3,1,2)
    plt.plot(Constants.WAVELENGTHS, eigenSpectra[1, :])
    plt.subplot(3,1,3)
    plt.plot(Constants.WAVELENGTHS, eigenSpectra[2, :])
    plt.xlabel("Wavelength (nm)")
    plt.show()

    plt.clf()
    plt.subplot(3,1,1)
    plt.plot(Constants.WAVELENGTHS, eigenSpectra[3, :])
    plt.title("Principle Components 4 - 6")
    plt.subplot(3,1,2)
    plt.plot(Constants.WAVELENGTHS, eigenSpectra[4, :])
    plt.subplot(3,1,3)
    plt.plot(Constants.WAVELENGTHS, eigenSpectra[5, :])
    plt.xlabel("Wavelength (nm)")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run linear regression on the wavelengths')
    parser.add_argument('date', type=str, nargs=1,
                         help='Data collection date YYYY_MMDD')
    parser.add_argument('-s', '--subset', default=False, action='store_true',
                         help="Plot only a random subset of the total dataset")

    args = parser.parse_args()
    main(args.date[0], args.subset)