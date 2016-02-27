# Written by Andrew Donelick
# andrew.donelick@msu.montana.edu
# Montana State University - Optical Remote Sensing Lab


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
