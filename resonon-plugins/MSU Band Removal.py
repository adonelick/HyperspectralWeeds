# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 9 February 2016
# Montana State University - Optical Remote Sensing Lab

"""
This plugin removes specified bands from a given datacube. Unlike the provided
plugin from Resonon, which simply deletes bands you wish to remove, this
plugin replaces the values in the bands you wish to remove with NaN values.
This way, it is clear that you are ignoring certain wavelengths when you 
are looking at plots of spectra.
""" 

from spectronon.workbench.plugin import CubePlugin
from resonon.utils.spec import SpecWavelength
from resonon.core.data import util
import numpy as np

class MSUBandRemoval(CubePlugin):
    """
    Replaces a specified range of bands with NaN values.
    """

    label = "MSU Band Removal" 
    defaultRenderer = "TriBand"
    userLevel = 1

    def setup(self):
        """
        Allow the user to select the range of bands he/she wishes to remove.

        :return: (None)
        """

        self.minBand  = SpecWavelength(label="Min Wavelength to Remove",
                                       datacube = self.datacube)
        self.maxBand  = SpecWavelength(label="Max Wavelength to Remove",
                                       datacube = self.datacube)

    def action(self):
        """
        Replaces the values in the specified range of wavelengths with NaNs

        :return: (Spectronon DataCube) new datacube with bands replaced with NaNs
        """

        # Make a new cube, which will contain the processed spectral bands
        newcube = util.makeEmptyCube(mode="memory",
                                     typechar='f',
                                     rotationString=self.datacube.getRotationString())

        bands = self.datacube.getBandCount()
        samples = self.datacube.getSampleCount()
        lines = self.datacube.getLineCount()
        wavelengths = self.datacube.getWavelengthList()

        minBand = self.minBand.value
        maxBand = self.maxBand.value

        self.wb.postMessage(str(minBand))

        for wavelength in wavelengths:

            currentBand = self.datacube.getBandAtWavelength(wavelength)
            currentBand = currentBand.astype(np.float32)
            if (minBand <= wavelength <= maxBand):
                currentBand[:,:] = np.NaN

            newcube.appendBandWithWavelength(currentBand, wavelength)

        return newcube