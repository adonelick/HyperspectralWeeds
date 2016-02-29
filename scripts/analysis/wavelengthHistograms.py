# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# Montana State University - Optical Remote Sensing Lab

"""
This script is used to visualize portions of the data by allowing
the user to plot histograms of specified wavelengths in collected data.

Usage:
    
    python wavelengthHistograms.py [date] [-w wavelengths] [-k keywords] [-l]

    date: Data collection data (ex: 2015_1211)
    wavelengths: Wavelengths in nm you wish to plot (ex: 600)
    keywords: Strings in the filenames to be included in plots
    leaves: Plot only a single point per leaf vs. all spectra in a leaf

"""

import argparse
import os
import sys
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

sys.path.append("..")
from common import FileIO
from common.Constants import *
from common.WavelengthCalculations import wavelengthToIndex

BINS = 50


def main(date, wavelengths, plotLeaves, keywords=[]):
    """
    Plot the histogram of a specified list of wavelengths.

    :param date: (string) Data collection date YYYY_MMDD
    :param wavelengths: (list) Wavelengths to plot histograms

    :return: (None)
    """

    numHistograms = len(wavelengths)
    wavelengthIndices = map(wavelengthToIndex, wavelengths)


    # Get the data files we will be looking at
    dataPath = DATA_PATHS[date]
    filesToPlot = FileIO.getDatafileNames(dataPath, keywords)

    pointsDR = np.zeros((1, numHistograms))
    pointsGR = np.zeros((1, numHistograms))
    pointsSUS = np.zeros((1, numHistograms))

    for name in filesToPlot:

        tokens = name[0:-4].split('_')
        map(lambda x: x.lower(), tokens)

        plant = tokens[0]
        resistance = tokens[1]
        imageType = tokens[2]
        index = int(tokens[4])

        filePath = os.path.join(dataPath, name)
        data = FileIO.loadCSV(filePath)

        # Extract the relevant data from the spectra in the data file
        histogramData = data[:, wavelengthIndices]

        if plotLeaves:

            meanLeaf = map(lambda i: np.mean(histogramData[:,i]), xrange(0, numHistograms))

            if resistance == SUSCEPTIBLE:
                pointsSUS = np.append(pointsSUS, [meanLeaf], axis=0)
            elif resistance == DR_RESISTANT:
                pointsDR = np.append(pointsDR, [meanLeaf], axis=0)
            elif resistance == GR_RESISTANT:
                pointsGR = np.append(pointsGR, [meanLeaf], axis=0)
            else:
                raise Exception("Unknown resistance type: " + resistance)

        else:

            if resistance == SUSCEPTIBLE:
                pointsSUS = np.append(pointsSUS, histogramData, axis=0)
            elif resistance == DR_RESISTANT:
                pointsDR = np.append(pointsDR, histogramData, axis=0)
            elif resistance == GR_RESISTANT:
                pointsGR = np.append(pointsGR, histogramData, axis=0)
            else:
                raise Exception("Unknown resistance type: " + resistance)

    # Ignore the top row of zeros
    pointsDR = pointsDR[1:,:]
    pointsGR = pointsGR[1:,:]
    pointsSUS = pointsSUS[1:,:]

    # Plot the histograms of the given wavelengths
    for i, wavelength in enumerate(wavelengths):

        x = pointsSUS[:, i]
        y = pointsDR[:, i]
        z = pointsGR[:, i]

        if np.NaN in x:
            print "Skipping wavelength", wavelength
            break

        meanSUS = np.mean(x)
        meanDR = np.mean(y)
        meanGR = np.mean(z)

        stdSUS = np.std(x)
        stdDR = np.std(y)
        stdGR = np.std(z)
        
        print "Mean susceptible reflectance:", meanSUS
        print "Mean glyphosate reflectance:", meanGR
        print "Mean dicamba reflectance:", meanDR, '\n'

        print "Susceptible reflectance standard deviation:", stdSUS
        print "Glyphosate reflectance standard deviation:", stdGR
        print "Dicamba reflectance standard deviation:", stdDR, '\n'

        fValue, pValue = stats.f_oneway(x, y, z)
        print "One way ANOVA p-Value:", pValue

        print "t-test between susceptible and glyphosate:"
        print "  ", stats.ttest_ind(x, z)
        print "t-test between susceptible and dicamba:"
        print "  ", stats.ttest_ind(x, y)
        print "t-test between dicamba and glyphosate:"
        print "  ", stats.ttest_ind(y, z)


        plt.hist(x, BINS, alpha=0.5, label=RESISTANCE_STRINGS[SUSCEPTIBLE])
        plt.hist(y, BINS, alpha=0.5, label=RESISTANCE_STRINGS[DR_RESISTANT])
        plt.hist(z, BINS, alpha=0.5, label=RESISTANCE_STRINGS[GR_RESISTANT])
        plt.title('Reflectance Distribution for Wavelength ' + str(wavelength) + ' nm')
        plt.xlabel('Reflectance')
        plt.ylabel('Frequency')
        plt.legend(loc='upper left')
        plt.show()




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot wavelength histograms')
    parser.add_argument('date', type=str, nargs=1,
                         help='Data collection date YYYY_MMDD')
    parser.add_argument('-w', '--wavelengths', default=WAVELENGTHS, type=float, nargs='*',
                         help="Wavelengths to plot")
    parser.add_argument('-l', '--leaves', default=False, action='store_true',
                         help="One leaf -> one sample")
    parser.add_argument('-k', '--keywords', default=[], type=str, nargs='*',
                         help="Filename keywords to include")
    args = parser.parse_args()

    main(args.date[0], args.wavelengths, args.leaves, args.keywords)