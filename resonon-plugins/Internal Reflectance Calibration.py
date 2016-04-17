# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 16 April 2016
# Montana State University - Optical Remote Sensing Lab

"""
This plugin allows for calibration of the collected data (brightness values)
to reflectance. For this plugin to work, it is required to have a 
white target (such as a Spectralon panel) in the field of view of the
current datacube.
"""

from spectronon.workbench.plugin import SelectPlugin
from resonon.utils.spec import SpecBool
from resonon.core.data import util
from MSU_Constants import SPECTRALON_REFLECTANCE
from MSU_Constants import MIN_SPECTRALON_WAVELENGTH 
from MSU_Constants import MAX_SPECTRALON_WAVELENGTH
from MSU_Constants import SPECTRALON_WAVELENGTH_STEP
from MSU_Constants import BANDS_TO_REMOVE

import numpy as np


class InternalReflectanceCalibration(SelectPlugin):
    """Calibrate the spectra to reflectance with an internal target""" 
    label = "Internal Reflectance Calibration"
    userLevel = 1


    def setup(self):
        """
        Set up the calibration process. Right now, we only check if the user
        wants to remove some spectral bands from the calibrated datacube.

        :return: (None)
        """

        message = "Remove bad spectral bands?"
        self.removeBands = SpecBool(message, defaultValue=False)


    def action(self):
        """
        Calculates the correction gain to apply to all spectra in the dataCube
        to transform them to reflectance spectra. This function assumes that
        the user has selected a region of the Spectralon reflectance target
        to use for the calibration.

        :return: (None)
        """

        # Retrieve useful information about the datacube and the user selection
        dataCube = self.datacube.getArray(asBIP=True)
        lines, samples, bands = dataCube.shape
        datacubeName = self.datacube.getName()
        pointList = self.pointlist
        wavelengths = self.datacube.getWavelengthList()

        # Extract the region of interest from the selected points
        pointLines = map(lambda x: x[1], pointList)
        pointSamples = map(lambda x: x[0], pointList)

        minLine = min(pointLines)
        maxLine = max(pointLines)
        minSample = min(pointSamples)
        maxSample = max(pointSamples)

        # Extract the selected data, fetch the spectralon reflectance data        
        data = dataCube[minLine:maxLine+1, minSample:maxSample+1, :]
        reflectances = np.array(map(getSpectralonReflectance, wavelengths))
        assert(maxSample - minSample + 1 == samples, "You did not select a large enough ROI!")

        # Calculate the correction gains to apply to the rest of the spectra
        corrections = np.zeros((samples, len(wavelengths)), dtype=np.float32)
        for i in xrange(samples):

            mean = np.mean(data[:, i, :], axis=0).astype(self.datacube.getDType()).flatten()
            corrections[i, :] = 1.0 * reflectances / mean

        # Apply the correction to all of the spectra in the datacube
        reflectanceCube = np.zeros_like(dataCube, dtype=np.float32)
        for i in xrange(lines):
            for j in xrange(samples):
                reflectanceCube[i, j, :] = corrections[j, :] * dataCube[i, j, :]

        # If desired, remove the spectral bands from the calibrated datacube
        if self.removeBands.value:
            for removal in BANDS_TO_REMOVE:
                lowerLimit = removal[0]
                upperLimit = removal[1]

                for i, wavelength in enumerate(wavelengths):
                    if (lowerLimit <= wavelength <= upperLimit):
                        reflectanceCube[:,:,i] = np.NaN

        calibratedDatacube = createNewDatacube(self.wb, reflectanceCube,
                                               wavelengths,
                                               self.datacube.getRotationString())

        # Place the new, calibrated datacube on the workbench for saving,
        # manipulation, or whatever you want to do with it
        newName = datacubeName + "_Reflectance"
        self.wb.addCube(calibratedDatacube, name=newName)


def getSpectralonReflectance(wavelength):
    """
    Calculates the reflectance of the Spectralon panel for any 
    given wavelength between 300 and 1500 nm. If the wavelength
    exceeds these limits, the either the reflectance at the min 
    or max wavelength present will be returned.

    :param wavelength: (float) wavelength in nanometers

    :return: (float) reflectance of the Spectralon at that wavelength
    """
    
    lower = int(wavelength - (wavelength % SPECTRALON_WAVELENGTH_STEP))
    upper = lower + SPECTRALON_WAVELENGTH_STEP

    if lower <= MIN_SPECTRALON_WAVELENGTH:
        return SPECTRALON_REFLECTANCE[MIN_SPECTRALON_WAVELENGTH]

    if upper >= MAX_SPECTRALON_WAVELENGTH:
        return SPECTRALON_REFLECTANCE[MAX_SPECTRALON_WAVELENGTH]

    upperReflectance = SPECTRALON_REFLECTANCE[upper]
    lowerReflectance = SPECTRALON_REFLECTANCE[lower]

    slope = 1.0 * (upperReflectance - lowerReflectance) / SPECTRALON_WAVELENGTH_STEP
    return slope * (wavelength - lower) + lowerReflectance


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

