# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 25 March 2016
# Montana State University - Optical Remote Sensing Lab


import numpy as np
import os
import sys
sys.path.append("..")
import mkl
import argparse
from sklearn.decomposition import IncrementalPCA

import plotly.plotly as py
import plotly.graph_objs as go

from scripts.common import FileIO
from scripts.common import Constants


NUM_SAMPLES = 1000



def main(date):

    mkl.set_num_threads(8)

    # Load the training and testing data into memory
    trainX, trainY = FileIO.loadTrainingData(date)

    indices = np.random.choice(range(0, len(trainY)), size=NUM_SAMPLES, replace=False)
    X = trainX[indices,:]
    y = trainY[indices]

    X = np.nan_to_num(X)

    # Break the data into resistance classes
    susIndex = Constants.LABEL_TO_INDEX[Constants.SUSCEPTIBLE]
    drIndex = Constants.LABEL_TO_INDEX[Constants.DR_RESISTANT]
    grIndex = Constants.LABEL_TO_INDEX[Constants.GR_RESISTANT]

    susX = X[y==susIndex, :]
    drX = X[y==drIndex, :]
    grX = X[y==grIndex, :]

    pca = IncrementalPCA(n_components=3)

    pointsSUS = pca.fit_transform(susX)
    pointsGR= pca.fit_transform(grX)
    pointsDR = pca.fit_transform(drX)


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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run linear regression on the wavelengths')
    parser.add_argument('date', type=str, nargs=1,
                         help='Data collection date YYYY_MMDD')

    args = parser.parse_args()
    main(args.date[0])