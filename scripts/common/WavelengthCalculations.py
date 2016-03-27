# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# Montana State University - Optical Remote Sensing Lab

"""
This file contains useful helper functions for calculations 
involving wavelengths, wavelength conversions, etc.
"""

from Constants import *
import math


def wavelengthToIndex(wavelength):
    """
    Gets the index of the given wavelength in the wavelengths list.
    If the wavelength is not contained in the list, the function 
    will return the index of the wavelength closest to that which
    was given.

    :param wavelength: (float) wavelength in nanometers

    :return: (int) index of the wavelength
    """

    error = map(lambda x: math.fabs(x - wavelength), WAVELENGTHS)
    return error.index(min(error))


def indexToWavelength(index):
    """
    Gets the wavelength at a given index.

    :param index: (int) Index of desired wavelength

    :return: (float) Wavelength in nanometers
    """

    return WAVELENGTHS[index]


def wavelengthRegionToIndices(wavelength, width):
    """
    Converts a selected region, centered on wavelength and with a
    specified width, to a list of wavelenght indices.

    :param wavelength: (float) wavelength in nanometers
    :param width: (float) width of selected region in nanometers

    :return: (list of ints) 
    """

    halfWidth = width / 2.0
    maxWavelength = wavelength + halfWidth
    minWavelength = wavelength - halfWidth

    maxIndex = wavelengthToIndex(maxWavelength)
    minIndex = wavelengthToIndex(minWavelength)

    return range(minIndex, maxIndex + 1)