# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 26 March 2016
# Montana State University - Optical Remote Sensing Lab

"""
This script is used to visualize portions of the data by allowing
the user to plot three features against one another in 3d space. Because
this project focuses on hyperspectral data, the script allows the user
to specify which wavelengths should be plotted.

Usage:
    
    python plotWavelengths.py [date] [wavelength1] [wavelength2] [wavelength3] [-k keywords] [-a allSpectra]

    date: Data collection data (ex: 2015_1211)
    wavelength: Wavelengths in nm you wish to plot (ex: 600)
    keywords: Strings which should be included in the 
              filenames of files being plotted
    allSpectra: Determines where there is one point for every spectra
                collected, or one point for every leaf file
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

def main(date, wavelengths, keywords=[], allSpectra=False):
    """
    Plot three wavelengths against each other from a specified set of data.

    :param date: (string) Data collection date YYYY_MMDD
    :param wavelengths: (3-tuple) Wavelengths to plot against another
    :param keywords: (list of strings) Strings which should be included in the 
                                       filenames of files being plotted
    :allSpectra: (boolean) Determines where there is one point for every spectra
                           collected, or one point for every leaf file

    :return: (None)
    """

    # Convert the wavelengths to indices for accessing the data
    wavelengthIndices = map(wavelengthToIndex, wavelengths)
    wavelengthIndex1 = wavelengthIndices[0]
    wavelengthIndex2 = wavelengthIndices[1]
    wavelengthIndex3 = wavelengthIndices[2]


    # Get the data files we will be looking at
    dataPath = DATA_DIRECTORIES[date]
    filesToPlot = FileIO.getDatafileNames(dataPath, keywords)

    pointsDR = []
    pointsGR = []
    pointsSUS = []

    for name in filesToPlot:

        tokens = name[0:-4].split('_')
        map(lambda x: x.lower(), tokens)

        plant = tokens[0]
        resistance = tokens[1]

        filePath = os.path.join(dataPath, name)
        data = FileIO.loadCSV(filePath)

        if allSpectra:

            rows, columns = data.shape

            xValues = data[:, wavelengthIndex1]
            yValues = data[:, wavelengthIndex2]
            zValues = data[:, wavelengthIndex3]

            points = np.zeros((rows, 3))
            points[:, 0] = xValues
            points[:, 1] = yValues
            points[:, 2] = zValues
                
            if resistance == SUSCEPTIBLE:
                if pointsSUS == []:
                    pointsSUS = points
                else:
                    pointsSUS = np.append(pointsSUS, points, axis=0)

            elif resistance == DR_RESISTANT:
                if pointsDR == []:
                    pointsDR = points
                else:
                    pointsDR = np.append(pointsDR, points, axis=0)

            elif resistance == GR_RESISTANT:
                if pointsGR == []:
                    pointsGR = points
                else:
                    pointsGR = np.append(pointsGR, points, axis=0)
            else:
                raise Exception("Unknown resistance type: " + resistance)

        else:

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

    # Plot the wavelengths
    pointsDR = np.array(pointsDR)
    pointsGR = np.array(pointsGR)
    pointsSUS = np.array(pointsSUS)

    traceSUS = go.Scatter3d(
        x=pointsSUS[:, 0],
        y=pointsSUS[:, 1],
        z=pointsSUS[:, 2],
        mode='markers',
        name=RESISTANCE_STRINGS[SUSCEPTIBLE],
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
        name=RESISTANCE_STRINGS[DR_RESISTANT],
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
        name=RESISTANCE_STRINGS[GR_RESISTANT],
        marker=dict(
            size=5,
            line=dict(
                color='rgba(0, 0, 255, 0)',
                width=0.1
            ),
            opacity=0
        )
    )

    layout = go.Layout(
        title='3D Wavelength Plot',
        scene=go.Scene(
            xaxis=go.XAxis(title='Reflectance @ ' + str(wavelengths[0]) + ' nm'),
            yaxis=go.YAxis(title='Reflectance @ ' + str(wavelengths[1]) + ' nm'),
            zaxis=go.ZAxis(title='Reflectance @ ' + str(wavelengths[2]) + ' nm')
        )
    )

    data = [traceSUS, traceDR, traceGR]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='3D Wavelength Plot')




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot three wavelengths')
    parser.add_argument('date', type=str, nargs=1,
                         help='Data collection date YYYY_MMDD')
    parser.add_argument('wavelengths', type=float, nargs=3,
                         help="Wavelengths to plot")
    parser.add_argument('-k', '--keywords', default=[], type=str, nargs='*',
                         help="Filename keywords to include")
    parser.add_argument('-a', '--allSpectra', default=False, action='store_true',
                         help="Plot one point per spectrum in all datafiles")

    args = parser.parse_args()
    main(args.date[0], tuple(args.wavelengths), args.keywords, args.allSpectra)