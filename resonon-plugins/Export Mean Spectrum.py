# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 22 October 2015
# Montana State University - Optical Remote Sensing Lab

from spectronon.workbench.plugin import SelectPlugin

import numpy as np
import cPickle
import os

class ExportMeanSpectrum(SelectPlugin):
    """Export the mean spectrum of a region as a CSV file""" 
    label = "Export Mean Spectrum"
    userLevel = 1

    def action(self):
        """
        Saves the mean spectrum of a selected region as a CSV file.

        :return: (None)
        """

        # Retrieve useful information about the datacube and the user selection
        dataCube = self.datacube.getArray(asBIP=True)
        pointlist = self.pointlist
        wavelengths = self.datacube.getWavelengthList()

        # Calculate the mean spectrum of the selection
        data = dataCube[pointlist[:, 1], pointlist[:, 0], :]
        mean = np.mean(data, axis=0).astype(self.datacube.getTypeChar()).flatten()

        destination = self.wb.requestOpenFilename(message="Please Give a Destination for the Spectrum",
                                                  wildcard='*.csv')

        # Save the mean spectrum as a CSV file with the wavelength in the first
        # column, and the reflectance values in the second column.
        fileHandle = open(destination, 'wb')
        for i, wavelength in enumerate(wavelengths):
            fileHandle.write(str(wavelength))
            fileHandle.write(',')
            fileHandle.write(str(mean[i]))
            fileHandle.write('\n')

        fileHandle.close()

