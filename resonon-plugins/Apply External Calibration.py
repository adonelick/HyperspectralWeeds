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

class ApplyExternalCalibration(CubePlugin):
    """
    Applies an externally calculated correction factor to all spectra 
    within a datacube to convert the spectra to reflectances.
    """

    label = "Apply External Calibration" 
    defaultRenderer = "TriBand"
    userLevel = 1

    def setup(self):
        """
        Allow the user to select the file containing the correction which
        will be applied to all spectra in the datacube.

        :return: (None)
        """

        location = self.wb.requestOpenFilename(message="Please Give a Location for the Correction",
                                               wildcard='*.csv')
        self.correctionLocation = location


    def action(self):
        """
        Applies the correction to all spectra in the current datacube.

        :return: (Spectronon DataCube) new datacube calibrated to reflectance
        """

        dataCube = self.datacube.getArray(asBIP=True)
        datacubeName = self.datacube.getName()
        bands = self.datacube.getBandCount()
        samples = self.datacube.getSampleCount()
        lines = self.datacube.getLineCount()
        wavelengths = self.datacube.getWavelengthList()

        # Load the correction from disk
        correctionData = np.loadtxt(open(self.correctionLocation, "rb"),delimiter=",")
        correction = correctionData[:,1]
        assert(len(correction) == bands)

        # Apply the correction to all of the spectra in the datacube
        reflectanceCube = np.zeros_like(dataCube, dtype=np.float32)
        for i in xrange(lines):
            for j in xrange(samples):
                reflectanceCube[i, j, :] = correction * dataCube[i, j, :]

        calibratedDatacube = createNewDatacube(self.wb, reflectanceCube,
                                               wavelengths,
                                               self.datacube.getRotationString())

        # Place the new, calibrated datacube on the workbench for saving,
        # manipulation, or whatever you want to do with it
        newName = datacubeName + "_Reflectance"
        self.wb.addCube(calibratedDatacube, name=newName)


def createNewDatacube(workbench, data, wavelengths, rotationString):
    """
    Creates a Spectronon DataCube object from the supplied data matrix. 

    :param workbench: (Spectronon workbench) Reference to useful stuff in Spectronon
    :param data: (matrix of floats) Data which should be put into a cube
    :param wavelengths: (list of floats) Wavelengths of the datacube
    :param rotationString: (string) Rotation format for the datacube

    :return: (Spectronon DataCube) New datacube containing provided data
    """

    newcube = util.makeEmptyCube(mode="memory",
                                 typechar='f',
                                 rotationString=rotationString)

    for i, wavelength in enumerate(wavelengths):

        band = data[:,:, i]
        newcube.appendBandWithWavelength(band, wave=wavelength)

    return newcube
