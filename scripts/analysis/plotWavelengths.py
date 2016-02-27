# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 25 February 2016
# Montana State University - Optical Remote Sensing Lab

"""
This script is used to visualize portions of the data by allowing
the user to plot three features against one another in 3d space. Because
this project focuses on hyperspectral data, the script allows the user
to specify which wavelengths should be plotted.

Usage:
    
    python plotWavelengths.py [date] [wavelength1] [wavelength2] [wavelength3]

    date: Data collection data (ex: 2015_1211)
    wavelength: Wavelengths in nm you wish to plot (ex: 600)

"""

import argparse
import os
import sys
import numpy as np

import plotly.plotly as py
import plotly.graph_objs as go

sys.path.append("..")
from common import FileIO
from common.Constants import *
from common.WavelengthCalculations import wavelengthToIndex

def main(date, wavelengths, keywords=[]):
    """
    Plot three wavelengths against each other from a specified set of data.

    :param date: (string) Data collection date YYYY_MMDD
    :param wavelengths: (3-tuple) Wavelengths to plot against another

    :return: (None)
    """

    wavelengthIndices = map(wavelengthToIndex, wavelengths)
    wavelengthIndex1 = wavelengthIndices[0]
    wavelengthIndex2 = wavelengthIndices[1]
    wavelengthIndex3 = wavelengthIndices[2]


    # Get the data files we will be looking at
    dataPath = DATA_PATHS[date]
    filesToPlot = FileIO.getDatafileNames(dataPath, ["top"])

    pointsDR = []
    pointsGR = []
    pointsSUS = []

    for name in filesToPlot:

        tokens = name[0:-4].split('_')
        map(lambda x: x.lower(), tokens)

        plant = tokens[0]
        resistance = tokens[1]
        imageType = tokens[2]
        index = int(tokens[4])

        filePath = os.path.join(dataPath, name)
        data = FileIO.loadCSV(filePath)

        mean = np.mean(data, axis=0)
        meanValue1 = mean[wavelengthIndex1]
        meanValue2 = mean[wavelengthIndex2]
        meanValue3 = mean[wavelengthIndex3]

        if resistance == SUSCEPTIBLE:
            pointsSUS.append([meanValue1, meanValue2, meanValue3])
        elif resistance == DR_RESISTANT:
            pointsDR.append([meanValue1, meanValue2, meanValue3])
        elif resistance == GR_RESISTANT:
            pointsGR.append([meanValue1, meanValue2, meanValue3])
        else:
            raise Exception("Unknown resistance type: " + resistance)

    # Need method of determining wavelengths stored in each column

    pointsDR = np.array(pointsDR)
    pointsGR = np.array(pointsGR)
    pointsSUS = np.array(pointsSUS)

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
    py.iplot(fig, filename='3D Wavelength Plot')





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot three wavelengths')
    parser.add_argument('date', type=str, nargs=1,
                         help='Data collection date YYYY_MMDD')
    parser.add_argument('wavelengths', type=float, nargs=3,
                         help="Wavelengths to plot")
    parser.add_argument('-k', '--keywords', default=[], type=str, nargs='*',
                         help="Filename keywords to include")

    args = parser.parse_args()
    main(args.date[0], tuple(args.wavelengths), args.keywords)