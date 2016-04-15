# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# 9 February 2016
# Montana State University - Optical Remote Sensing Lab

"""
This plugin allows for calibration of the collected data (brightness values)
to reflectance. With this plugin, calibration spectra are collected from
a datacube of a Spectralon panel. The correction factor is computed from 
these spectra, which is then saved to disk. Another plugin can be used
to apply this correction to other datacubes.
"""

from spectronon.workbench.plugin import SelectPlugin
from MSU_Constants import SPECTRALON_REFLECTANCE
from MSU_Constants import MIN_SPECTRALON_WAVELENGTH 
from MSU_Constants import MAX_SPECTRALON_WAVELENGTH
from MSU_Constants import SPECTRALON_WAVELENGTH_STEP

import numpy as np

class ExportReflectanceCalibration(SelectPlugin):
    """Calibrate the spectra to reflectance with an internal target""" 
    label = "Export Reflectance Calibration"
    userLevel = 1

    def action(self):
        """
        Calculates the correction gain to apply to all spectra in the dataCube
        to transform them to reflectance spectra. This function assumes that
        the user has selected a region of the Spectralon reflectance target
        to use for the calibration. Saves the correction to disk.

        :return: (None)
        """

        # Retrieve useful information about the datacube and the user selection
        dataCube = self.datacube.getArray(asBIP=True)
        pointlist = self.pointlist
        wavelengths = self.datacube.getWavelengthList()

        # Calculate the mean spectrum of the selection. This will be the mean
        # spectrum of the Spectralon target panel
        data = dataCube[pointlist[:, 1], pointlist[:, 0], :]
        lines, samples, bands = dataCube.shape
        mean = np.mean(data, axis=0).astype(self.datacube.getDType()).flatten()

        # Calculate the correction gain to apply to the rest of the spectra
        reflectances = np.array(map(getSpectralonReflectance, wavelengths))
        correction = reflectances / mean

        # Save the mean correction as a CSV file with the wavelength in the first
        # column, and the correction values in the second column.
        destination = self.wb.requestOpenFilename(message="Please Give a Destination for the Correction",
                                                  wildcard='*.csv')

        fileHandle = open(destination, 'wb')
        for i, wavelength in enumerate(wavelengths):
            fileHandle.write(str(wavelength))
            fileHandle.write(',')
            fileHandle.write(str(correction[i]))
            fileHandle.write('\n')

        fileHandle.close()


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
