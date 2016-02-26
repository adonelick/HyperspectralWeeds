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
    wavelength: Wavelengths you wish to plot (ex: 600)

"""

import argparse
sys.path.append("..")
from common import FileIO
from common import Constants


def main(date, wavelengths):
    """
    Plot three wavelengths against each other from a specified set of data.

    :param date: (string) Data collection date YYYY_MMDD
    :param wavelengths: (3-tuple) Wavelengths to plot against another

    :return: (None)
    """

    # Get the data files we will be looking at
    dataPath = Constants.DATA_PATHS[date]
    filesToPlot = FileIO.getDatafileNames(dataPath)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot three wavelengths')
    parser.add_argument('date', type=str, nargs=1,
                         help='Data collection date YYYY_MMDD')
    parser.add_argument('wavelengths', type=float, nargs=3,
                         help="Wavelengths to plot")

    args = parser.parse_args()
    main(args.date, tuple(args.wavelengths))